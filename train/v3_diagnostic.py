"""Time-boxed v3 learnability and one-factor diagnostic runs.

Every invocation writes one self-describing run directory.  Runs using
LayerNorm are labelled ``diagnostic_non_g1``; LayerNorm-free runs are labelled
``diagnostic_g1_compatible`` because the widened internal dimensions can embed
the constructed weights with zeros in every added coordinate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional

from bf_transformer import Config
from bf_transformer.model import BFTransformer
from bf_transformer.registers import RegisterMap
from bf_transformer.verify import random_graph
from train.data import Example, make_example
from train.v3_model import V3BFTransformer, cosine_multiplier


@dataclass(frozen=True)
class DiagnosticRecipe:
    n_max: int
    train_n_max: int
    width: int | None
    layer_norm: bool
    supervision: str
    data_mode: str
    steps: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    minimum_lr_ratio: float
    validation_interval: int
    seed: int
    data_seed: int
    init_gain: float
    initialisation: str
    noise_std: float


@dataclass(frozen=True)
class DiagnosticBatch:
    streams: torch.Tensor
    targets: torch.Tensor
    node_mask: torch.Tensor
    source_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "DiagnosticBatch":
        return DiagnosticBatch(
            streams=self.streams.to(device),
            targets=self.targets.to(device),
            node_mask=self.node_mask.to(device),
            source_mask=self.source_mask.to(device),
        )


def collate_diagnostic(
    examples: list[Example],
    cfg: Config,
) -> DiagnosticBatch:
    if not examples:
        raise ValueError("cannot collate an empty diagnostic batch")
    max_nodes = max(example.graph.n for example in examples)
    targets = torch.full(
        (len(examples), cfg.rounds + 1, max_nodes),
        cfg.inf_b,
        dtype=torch.float32,
    )
    node_mask = torch.zeros(len(examples), max_nodes, dtype=torch.bool)
    source_mask = torch.zeros(len(examples), max_nodes, dtype=torch.bool)
    for row, example in enumerate(examples):
        n = example.graph.n
        targets[row, :, :n] = example.targets
        node_mask[row, :n] = True
        source_mask[row, example.source] = True
    return DiagnosticBatch(
        streams=torch.stack([example.stream for example in examples]),
        targets=targets,
        node_mask=node_mask,
        source_mask=source_mask,
    )


class GraphGenerator:
    """Deterministic v1-distribution graph source, fresh or fixed."""

    def __init__(
        self,
        cfg: Config,
        *,
        train_n_max: int,
        seed: int,
        data_mode: str,
        batch_size: int,
        fixed_size: int = 4096,
    ) -> None:
        self.cfg = cfg
        self.train_n_max = train_n_max
        self.rng = random.Random(seed)
        self.data_mode = data_mode
        self.batch_size = batch_size
        self.accepted = 0
        self.fixed: list[Example] = []
        self.fixed_order: list[int] = []
        self.fixed_cursor = 0
        if data_mode == "fixed":
            self.fixed = [self._new_example() for _ in range(fixed_size)]
        elif data_mode != "fresh":
            raise ValueError(f"unknown data mode: {data_mode!r}")

    def _new_example(self) -> Example:
        generator_cfg = (
            self.cfg
            if self.accepted % 2 == 0
            else Config(
                n_max=self.cfg.n_max,
                d_max=2,
                w_max=self.cfg.w_max,
                c_hard=self.cfg.c_hard,
            )
        )
        n = self.rng.randint(2, self.train_n_max)
        graph, source = random_graph(
            self.rng,
            cfg=generator_cfg,
            n=n,
            integer_weights=True,
            allow_negative=False,
        )
        self.accepted += 1
        return make_example(graph, source, self.cfg)

    def next(self) -> list[Example]:
        if self.data_mode == "fresh":
            return [self._new_example() for _ in range(self.batch_size)]
        if self.fixed_cursor + self.batch_size > len(self.fixed_order):
            self.fixed_order = list(range(len(self.fixed)))
            self.rng.shuffle(self.fixed_order)
            self.fixed_cursor = 0
        indices = self.fixed_order[
            self.fixed_cursor : self.fixed_cursor + self.batch_size
        ]
        self.fixed_cursor += self.batch_size
        return [self.fixed[index] for index in indices]


def generate_evaluation_set(
    cfg: Config,
    *,
    train_n_max: int,
    seed: int,
    count: int,
) -> list[Example]:
    source = GraphGenerator(
        cfg,
        train_n_max=train_n_max,
        seed=seed,
        data_mode="fresh",
        batch_size=count,
    )
    return source.next()


def distance_register(
    state: torch.Tensor,
    *,
    node_count: int,
    cfg: Config,
) -> torch.Tensor:
    index = RegisterMap(cfg).idx("DIST")
    return state[..., 1 : 1 + node_count, index]


def diagnostic_loss(
    states: list[torch.Tensor],
    batch: DiagnosticBatch,
    *,
    supervision: str,
    cfg: Config,
) -> torch.Tensor:
    node_count = batch.targets.shape[-1]
    if supervision == "final":
        predictions = distance_register(
            states[-1],
            node_count=node_count,
            cfg=cfg,
        ).unsqueeze(1)
        targets = batch.targets[:, -1:].to(predictions.device)
        mask = batch.node_mask[:, None, :].to(predictions.device)
    elif supervision == "round":
        predictions = torch.stack(
            [
                distance_register(state, node_count=node_count, cfg=cfg)
                for state in states[1:]
            ],
            dim=1,
        )
        targets = batch.targets[:, 1:].to(predictions.device)
        mask = batch.node_mask[:, None, :].to(predictions.device).expand_as(
            predictions
        )
    else:
        raise ValueError(f"unknown supervision: {supervision!r}")
    losses = functional.smooth_l1_loss(
        predictions / cfg.inf_b,
        targets / cfg.inf_b,
        beta=0.01,
        reduction="none",
    )
    return losses[mask].mean()


def diagnostic_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    node_mask: torch.Tensor,
    source_mask: torch.Tensor,
    *,
    cfg: Config,
) -> dict[str, float]:
    prediction = prediction.detach()
    target = target.to(prediction.device)
    node_mask = node_mask.to(prediction.device)
    source_mask = source_mask.to(prediction.device)
    unreachable_target = target >= cfg.inf_b - 0.5
    unreachable_prediction = prediction >= cfg.inf_b - 0.5
    exact = torch.where(
        unreachable_target,
        unreachable_prediction,
        (prediction - target).abs() < 0.5,
    )
    exact = exact & node_mask
    graph_exact = (
        (exact | ~node_mask).all(dim=-1).to(torch.float32).mean().item()
    )
    node_exact = exact.sum().item() / node_mask.sum().item()
    reachable_non_source = node_mask & ~unreachable_target & ~source_mask
    if reachable_non_source.any():
        reachable_non_source_exact = (
            exact[reachable_non_source].to(torch.float32).mean().item()
        )
    else:
        reachable_non_source_exact = math.nan
    reachability_correct = (
        (unreachable_prediction == unreachable_target) & node_mask
    )
    reachability_accuracy = (
        reachability_correct.sum().item() / node_mask.sum().item()
    )
    bounded_prediction = torch.where(
        unreachable_prediction,
        torch.full_like(prediction, cfg.inf_b),
        prediction,
    )
    errors = (bounded_prediction - target).abs()
    finite_mask = node_mask & ~unreachable_target
    return {
        "graph_exact": graph_exact,
        "node_exact": node_exact,
        "reachable_non_source_exact": reachable_non_source_exact,
        "reachability_accuracy": reachability_accuracy,
        "mae": errors[node_mask].mean().item(),
        "finite_mae": errors[finite_mask].mean().item(),
        "count_graphs": float(node_mask.shape[0]),
        "count_nodes": float(node_mask.sum().item()),
        "count_reachable_non_source": float(reachable_non_source.sum().item()),
    }


def _batches(examples: list[Example], size: int):
    for start in range(0, len(examples), size):
        yield examples[start : start + size]


@torch.no_grad()
def evaluate(
    model: V3BFTransformer,
    examples: list[Example],
    *,
    cfg: Config,
    supervision: str,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    all_prediction: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    all_node_mask: list[torch.Tensor] = []
    all_source_mask: list[torch.Tensor] = []
    loss_numerator = 0.0
    loss_denominator = 0
    maximum_nodes = max(example.graph.n for example in examples)
    for group in _batches(examples, batch_size):
        batch = collate_diagnostic(group, cfg).to(device)
        states = model.forward_with_states(batch.streams)
        loss = diagnostic_loss(
            states,
            batch,
            supervision=supervision,
            cfg=cfg,
        )
        loss_numerator += loss.item() * len(group)
        loss_denominator += len(group)
        prediction = distance_register(
            states[-1],
            node_count=batch.targets.shape[-1],
            cfg=cfg,
        ).cpu()
        target = batch.targets[:, -1].cpu()
        padded_prediction = torch.zeros(len(group), maximum_nodes)
        padded_target = torch.zeros(len(group), maximum_nodes)
        padded_node_mask = torch.zeros(
            len(group),
            maximum_nodes,
            dtype=torch.bool,
        )
        padded_source_mask = torch.zeros_like(padded_node_mask)
        width = prediction.shape[1]
        padded_prediction[:, :width] = prediction
        padded_target[:, :width] = target
        padded_node_mask[:, :width] = batch.node_mask.cpu()
        padded_source_mask[:, :width] = batch.source_mask.cpu()
        all_prediction.append(padded_prediction)
        all_target.append(padded_target)
        all_node_mask.append(padded_node_mask)
        all_source_mask.append(padded_source_mask)
    metrics = diagnostic_metrics(
        torch.cat(all_prediction),
        torch.cat(all_target),
        torch.cat(all_node_mask),
        torch.cat(all_source_mask),
        cfg=cfg,
    )
    metrics["supervised_loss"] = loss_numerator / loss_denominator
    if was_training:
        model.train()
    return metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _checkpoint_key(metrics: dict[str, float], step: int) -> tuple[float, ...]:
    """Exclude the sentinel/source identity baseline from primary selection."""
    return (
        metrics["reachable_non_source_exact"],
        metrics["graph_exact"],
        metrics["node_exact"],
        -metrics["finite_mae"],
        -float(step),
    )


def _parameter_distance(
    model: V3BFTransformer,
    reference: V3BFTransformer,
) -> float:
    squared = 0.0
    with torch.no_grad():
        for current, target in zip(
            model.parameters(),
            reference.parameters(),
            strict=True,
        ):
            squared += (current - target).pow(2).sum().item()
    return math.sqrt(squared)


def run_diagnostic(
    recipe: DiagnosticRecipe,
    *,
    output_dir: Path,
    device_name: str,
) -> dict:
    if recipe.train_n_max > recipe.n_max:
        raise ValueError("train_n_max cannot exceed n_max")
    cfg = Config(n_max=recipe.n_max)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    _seed_everything(recipe.seed)
    random_initialisation = recipe.initialisation == "random"
    model = V3BFTransformer(
        cfg,
        tied=True,
        width=recipe.width,
        layer_norm=recipe.layer_norm,
        random_initialisation=random_initialisation,
        init_gain=recipe.init_gain,
    ).to(device)
    constructed_reference: V3BFTransformer | None = None
    if recipe.initialisation == "constructed-noise":
        if not model.g1_compatible:
            raise ValueError("constructed-noise requires a G1-compatible model")
        constructed = BFTransformer(cfg)
        model.load_constructed_weights(constructed)
        constructed_reference = V3BFTransformer(
            cfg,
            tied=True,
            width=recipe.width,
            layer_norm=False,
            random_initialisation=False,
        ).to(device)
        constructed_reference.load_constructed_weights(constructed)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(torch.randn_like(parameter) * recipe.noise_std)
    elif recipe.initialisation != "random":
        raise ValueError(
            f"unknown initialisation: {recipe.initialisation!r}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "initial.pt")
    validation = generate_evaluation_set(
        cfg,
        train_n_max=recipe.train_n_max,
        seed=recipe.data_seed + 1,
        count=512,
    )
    test = generate_evaluation_set(
        cfg,
        train_n_max=recipe.train_n_max,
        seed=recipe.data_seed + 2,
        count=1024,
    )
    training_source = GraphGenerator(
        cfg,
        train_n_max=recipe.train_n_max,
        seed=recipe.data_seed,
        data_mode=recipe.data_mode,
        batch_size=recipe.batch_size,
    )
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=lambda step: cosine_multiplier(
            step,
            total_steps=recipe.steps,
            warmup_steps=recipe.warmup_steps,
            minimum_ratio=recipe.minimum_lr_ratio,
        ),
    )
    history: list[dict] = []
    best_key: tuple[float, ...] | None = None
    best_step: int | None = None
    first_competent_step: int | None = None
    accumulated_loss = 0.0
    accumulated_steps = 0
    started = time.time()

    for step in range(recipe.steps + 1):
        if step % recipe.validation_interval == 0:
            metrics = evaluate(
                model,
                validation,
                cfg=cfg,
                supervision=recipe.supervision,
                device=device,
            )
            event = {
                "step": step,
                "validation": metrics,
                "mean_training_loss_since_previous": (
                    accumulated_loss / accumulated_steps
                    if accumulated_steps
                    else None
                ),
                "elapsed_seconds": time.time() - started,
            }
            if constructed_reference is not None:
                event["distance_from_constructed"] = _parameter_distance(
                    model,
                    constructed_reference,
                )
            history.append(event)
            print(json.dumps(event, sort_keys=True), flush=True)
            if (
                metrics["graph_exact"] >= 0.90
                and metrics["node_exact"] >= 0.98
                and first_competent_step is None
            ):
                first_competent_step = step
            if step > 0:
                key = _checkpoint_key(metrics, step)
                if best_key is None or key > best_key:
                    best_key = key
                    best_step = step
                    torch.save(model.state_dict(), output_dir / "best.pt")
            accumulated_loss = 0.0
            accumulated_steps = 0
        if step == recipe.steps:
            break

        batch = collate_diagnostic(training_source.next(), cfg).to(device)
        optimiser.zero_grad(set_to_none=True)
        if recipe.supervision == "round":
            states = model.forward_with_states(batch.streams)
        else:
            states = [model(batch.streams)]
        loss = diagnostic_loss(
            states,
            batch,
            supervision=recipe.supervision,
            cfg=cfg,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        accumulated_loss += loss.item()
        accumulated_steps += 1

    torch.save(model.state_dict(), output_dir / "final.pt")
    if best_step is None:
        raise RuntimeError("no post-update validation checkpoint was evaluated")
    model.load_state_dict(
        torch.load(
            output_dir / "best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    best_validation = evaluate(
        model,
        validation,
        cfg=cfg,
        supervision=recipe.supervision,
        device=device,
    )
    best_test = evaluate(
        model,
        test,
        cfg=cfg,
        supervision=recipe.supervision,
        device=device,
    )
    result = {
        "phase": "diagnostic",
        "g1_status": (
            "diagnostic_g1_compatible"
            if model.g1_compatible
            else "diagnostic_non_g1"
        ),
        "recipe": asdict(recipe),
        "parameter_count": model.parameter_count(),
        "best_step": best_step,
        "first_competent_step": first_competent_step,
        "competent": (
            best_validation["graph_exact"] >= 0.90
            and best_validation["node_exact"] >= 0.98
        ),
        "best_validation": best_validation,
        "best_test": best_test,
        "history": history,
        "elapsed_seconds": time.time() - started,
    }
    if constructed_reference is not None:
        result["initial_distance_from_constructed"] = history[0][
            "distance_from_constructed"
        ]
        result["final_distance_from_constructed"] = history[-1][
            "distance_from_constructed"
        ]
    _write_json(output_dir / "run.json", result)
    print(
        json.dumps(
            {
                "event": "V3_DIAGNOSTIC_COMPLETE",
                "output": str(output_dir),
                "g1_status": result["g1_status"],
                "best_step": best_step,
                "competent": result["competent"],
                "best_validation": best_validation,
                "best_test": best_test,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def parse_args() -> tuple[DiagnosticRecipe, Path, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-max", type=int, required=True)
    parser.add_argument("--train-n-max", type=int, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layer-norm", action="store_true")
    parser.add_argument("--supervision", choices=("round", "final"), default="round")
    parser.add_argument("--data-mode", choices=("fresh", "fixed"), default="fresh")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=20260730)
    parser.add_argument("--init-gain", type=float, default=0.1)
    parser.add_argument(
        "--initialisation",
        choices=("random", "constructed-noise"),
        default="random",
    )
    parser.add_argument("--noise-std", type=float, default=1e-3)
    args = parser.parse_args()
    recipe = DiagnosticRecipe(
        n_max=args.n_max,
        train_n_max=args.train_n_max,
        width=None if args.width == 0 else args.width,
        layer_norm=args.layer_norm,
        supervision=args.supervision,
        data_mode=args.data_mode,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        minimum_lr_ratio=args.minimum_lr_ratio,
        validation_interval=args.validation_interval,
        seed=args.seed,
        data_seed=args.data_seed,
        init_gain=args.init_gain,
        initialisation=args.initialisation,
        noise_std=args.noise_std,
    )
    return recipe, args.output, args.device


def main() -> None:
    recipe, output, device = parse_args()
    run_diagnostic(recipe, output_dir=output, device_name=device)


if __name__ == "__main__":
    main()
