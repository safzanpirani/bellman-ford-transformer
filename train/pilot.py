"""Registered losses, metrics, and execution for the 12-run pilot."""

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
from torch import nn

from bf_transformer.registers import RegisterMap

from .config import TRAIN_CONFIG
from .data import (
    Batch,
    Example,
    collate_examples,
    generate_registered_corpus,
    load_corpus,
    save_corpus,
)


TOTAL_STEPS = 4000
BATCH_SIZE = 32
VALIDATION_INTERVAL = 200
MODEL_SEEDS = (17, 29, 43)
CONDITIONS = {
    "tied-final": (True, "final"),
    "tied-round": (True, "round"),
    "untied-final": (False, "final"),
    "untied-round": (False, "round"),
}
_DISTANCE_INDEX = RegisterMap(TRAIN_CONFIG).idx("DIST")


def _distance_register(state: torch.Tensor, node_count: int) -> torch.Tensor:
    return state[..., 1 : 1 + node_count, _DISTANCE_INDEX]


def supervised_loss(
    states: list[torch.Tensor],
    batch: Batch,
    *,
    supervision: str,
) -> torch.Tensor:
    """Smooth-L1 loss exactly as frozen in the pre-registration."""
    node_count = batch.targets.shape[-1]
    if supervision == "final":
        predictions = _distance_register(states[-1], node_count).unsqueeze(1)
        targets = batch.targets[:, -1:].to(predictions.device)
        mask = batch.node_mask[:, None, :].to(predictions.device)
    elif supervision == "round":
        predictions = torch.stack(
            [
                _distance_register(state, node_count)
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
        predictions / TRAIN_CONFIG.inf_b,
        targets / TRAIN_CONFIG.inf_b,
        beta=0.01,
        reduction="none",
    )
    return losses[mask].mean()


def behaviour_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Registered node/graph exact match and bounded-error summaries."""
    prediction = prediction.detach()
    target = target.to(prediction.device)
    mask = mask.to(prediction.device)
    unreachable_target = target >= TRAIN_CONFIG.inf_b - 0.5
    exact = torch.where(
        unreachable_target,
        prediction >= TRAIN_CONFIG.inf_b - 0.5,
        (prediction - target).abs() < 0.5,
    )
    exact = exact & mask
    node_exact = exact.sum().item() / mask.sum().item()
    graph_exact = (
        (exact | ~mask).all(dim=-1).to(torch.float32).mean().item()
    )
    bounded_prediction = torch.where(
        prediction >= TRAIN_CONFIG.inf_b - 0.5,
        torch.full_like(prediction, TRAIN_CONFIG.inf_b),
        prediction,
    )
    errors = (bounded_prediction - target).abs()[mask]
    return {
        "node_exact": node_exact,
        "graph_exact": graph_exact,
        "mae": errors.mean().item(),
        "median_ae": errors.median().item(),
        "count_nodes": float(mask.sum().item()),
        "count_graphs": float(mask.shape[0]),
    }


def cosine_learning_rate(
    step: int,
    *,
    total_steps: int = 4000,
    warmup_steps: int = 200,
    minimum_ratio: float = 0.1,
) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine


@dataclass(frozen=True)
class Condition:
    name: str
    tied: bool
    supervision: str


class FixedBatchStream:
    """Infinite registered data order, identical across all model seeds."""

    def __init__(self, examples: list[Example]) -> None:
        self.examples = examples
        self.generator = torch.Generator().manual_seed(9001)
        self.order = torch.empty(0, dtype=torch.int64)
        self.cursor = 0

    def next(self) -> list[Example]:
        if self.cursor + BATCH_SIZE > len(self.order):
            self.order = torch.randperm(
                len(self.examples),
                generator=self.generator,
            )
            self.cursor = 0
        indices = self.order[self.cursor : self.cursor + BATCH_SIZE]
        self.cursor += BATCH_SIZE
        return [self.examples[index] for index in indices.tolist()]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _iter_batches(
    examples: list[Example],
    batch_size: int = 64,
):
    for start in range(0, len(examples), batch_size):
        yield examples[start : start + batch_size]


def _records_metrics(records: list[dict]) -> dict[str, float]:
    max_nodes = max(record["n"] for record in records)
    prediction = torch.zeros(len(records), max_nodes)
    target = torch.zeros(len(records), max_nodes)
    mask = torch.zeros(len(records), max_nodes, dtype=torch.bool)
    for row, record in enumerate(records):
        n = record["n"]
        prediction[row, :n] = torch.tensor(record["prediction"])
        target[row, :n] = torch.tensor(record["target"])
        mask[row, :n] = True
    return behaviour_metrics(prediction, target, mask)


def _stratified_metrics(records: list[dict]) -> dict:
    by_n = {
        str(n): _records_metrics(
            [record for record in records if record["n"] == n]
        )
        for n in sorted({record["n"] for record in records})
    }
    graph_hop_labels = ("0", "1", "2", "3", "4", "5+")
    graph_groups: dict[str, list[dict]] = {
        label: [] for label in graph_hop_labels
    }
    for record in records:
        finite = [hop for hop in record["hops"] if hop >= 0]
        maximum = max(finite)
        label = str(maximum) if maximum < 5 else "5+"
        graph_groups[label].append(record)
    by_graph_hop = {
        label: (
            _records_metrics(group)
            if group
            else {"count_graphs": 0.0, "count_nodes": 0.0}
        )
        for label, group in graph_groups.items()
    }

    node_groups: dict[str, list[tuple[float, float]]] = {
        label: []
        for label in ("0", "1", "2", "3", "4", "5+", "unreachable")
    }
    for record in records:
        for hop, prediction, target in zip(
            record["hops"],
            record["prediction"],
            record["target"],
        ):
            if hop < 0:
                label = "unreachable"
            elif hop < 5:
                label = str(hop)
            else:
                label = "5+"
            node_groups[label].append((prediction, target))
    by_node_hop = {}
    for label, pairs in node_groups.items():
        if not pairs:
            by_node_hop[label] = {"count_nodes": 0.0}
            continue
        prediction = torch.tensor([[pair[0] for pair in pairs]])
        target = torch.tensor([[pair[1] for pair in pairs]])
        mask = torch.ones_like(target, dtype=torch.bool)
        metrics = behaviour_metrics(prediction, target, mask)
        by_node_hop[label] = {
            key: value
            for key, value in metrics.items()
            if key != "graph_exact"
        }
    return {
        "overall": _records_metrics(records),
        "by_n": by_n,
        "by_graph_hop": by_graph_hop,
        "by_node_hop": by_node_hop,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    examples: list[Example],
    device: torch.device,
    *,
    include_records: bool,
) -> tuple[dict, list[dict]]:
    was_training = model.training
    model.eval()
    records: list[dict] = []
    for example_batch in _iter_batches(examples):
        batch = collate_examples(example_batch).to(device)
        final_state = model(batch.streams)
        prediction = _distance_register(
            final_state,
            batch.targets.shape[-1],
        ).cpu()
        final_target = batch.targets[:, -1].cpu()
        for row, example in enumerate(example_batch):
            n = example.graph.n
            records.append(
                {
                    "n": n,
                    "source": example.source,
                    "hops": list(example.hops),
                    "prediction": prediction[row, :n].tolist(),
                    "target": final_target[row, :n].tolist(),
                }
            )
    if was_training:
        model.train()
    metrics = _stratified_metrics(records)
    return metrics, records if include_records else []


def _checkpoint_key(metrics: dict, step: int) -> tuple[float, ...]:
    overall = metrics["overall"]
    return (
        overall["graph_exact"],
        overall["node_exact"],
        -overall["mae"],
        -float(step),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _configure_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("highest")
    return device


def run_condition_seed(
    condition: Condition,
    seed: int,
    corpus: dict[str, list[Example]],
    output_root: Path,
    device: torch.device,
) -> dict:
    from .model import TrainableBFTransformer

    run_dir = output_root / condition.name / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    model = TrainableBFTransformer(tied=condition.tied).to(device)
    torch.save(model.state_dict(), run_dir / "initial.pt")

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lr_lambda=cosine_learning_rate,
    )
    batches = FixedBatchStream(corpus["train"])
    history: list[dict] = []
    best_key: tuple[float, ...] | None = None
    best_step = 0
    started_at = time.time()

    for step in range(TOTAL_STEPS + 1):
        if step % VALIDATION_INTERVAL == 0:
            validation, _ = evaluate_model(
                model,
                corpus["validation"],
                device,
                include_records=False,
            )
            event = {
                "step": step,
                "validation": validation["overall"],
                "elapsed_seconds": time.time() - started_at,
            }
            history.append(event)
            print(
                json.dumps(
                    {
                        "condition": condition.name,
                        "seed": seed,
                        **event,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            key = _checkpoint_key(validation, step)
            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                torch.save(model.state_dict(), run_dir / "best.pt")
        if step == TOTAL_STEPS:
            break

        batch = collate_examples(batches.next()).to(device)
        optimiser.zero_grad(set_to_none=True)
        if condition.supervision == "round":
            states = model.forward_with_states(batch.streams)
        else:
            states = [model(batch.streams)]
        loss = supervised_loss(
            states,
            batch,
            supervision=condition.supervision,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss in {condition.name} seed {seed} step {step}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()

    model.load_state_dict(
        torch.load(
            run_dir / "best.pt",
            map_location=device,
            weights_only=True,
        )
    )
    validation, validation_records = evaluate_model(
        model,
        corpus["validation"],
        device,
        include_records=True,
    )
    id_test, id_records = evaluate_model(
        model,
        corpus["id_test"],
        device,
        include_records=True,
    )
    ood_test, ood_records = evaluate_model(
        model,
        corpus["ood_test"],
        device,
        include_records=True,
    )
    g2_pass = (
        validation["overall"]["graph_exact"] >= 0.90
        and validation["overall"]["node_exact"] >= 0.98
    )
    result = {
        "condition": asdict(condition),
        "seed": seed,
        "device": str(device),
        "best_step": best_step,
        "g2_pass": g2_pass,
        "history": history,
        "validation": validation,
        "id_test": id_test,
        "ood_test": ood_test,
        "records": {
            "validation": validation_records,
            "id_test": id_records,
            "ood_test": ood_records,
        },
        "elapsed_seconds": time.time() - started_at,
    }
    _write_json(run_dir / "run.json", result)
    print(
        json.dumps(
            {
                "event": "run_complete",
                "condition": condition.name,
                "seed": seed,
                "best_step": best_step,
                "g2_pass": g2_pass,
                "validation": validation["overall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _condition_gate(results: list[dict]) -> dict:
    return {
        "g2_valid": all(result["g2_pass"] for result in results),
        "seeds": {
            str(result["seed"]): {
                "g2_pass": result["g2_pass"],
                "best_step": result["best_step"],
                "validation": result["validation"]["overall"],
            }
            for result in results
        },
    }


def run_registered_pilot(
    output_root: Path,
    device_name: str,
) -> dict:
    data_dir = output_root / "data"
    if (data_dir / "corpus.json").exists():
        corpus = load_corpus(data_dir)
        manifest = json.loads((data_dir / "manifest.json").read_text())
    else:
        corpus = generate_registered_corpus()
        manifest = save_corpus(corpus, data_dir)
    device = _configure_device(device_name)
    gates = {}
    for name, (tied, supervision) in CONDITIONS.items():
        condition = Condition(name, tied, supervision)
        results = [
            run_condition_seed(
                condition,
                seed,
                corpus,
                output_root,
                device,
            )
            for seed in MODEL_SEEDS
        ]
        gates[name] = _condition_gate(results)
        _write_json(output_root / "gates.json", gates)
    paired_valid = any(
        gates[f"tied-{supervision}"]["g2_valid"]
        and gates[f"untied-{supervision}"]["g2_valid"]
        for supervision in ("final", "round")
    )
    summary = {
        "manifest": manifest,
        "gates": gates,
        "paired_g2_valid": paired_valid,
    }
    _write_json(output_root / "training-summary.json", summary)
    print(
        json.dumps(
            {
                "event": "PILOT_TRAINING_COMPLETE",
                "paired_g2_valid": paired_valid,
                "gates": gates,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_registered_pilot(args.output, args.device)


if __name__ == "__main__":
    main()
