"""Additive analysis for the v4 saturation and drift diagnostics.

Training continues to use :mod:`train.v3_diagnostic` so the v3 task, loss,
metrics, data distribution, and model construction remain untouched.  This
module adds the two measurements v4 needs:

* an explicit, pre-stated saturation test over two trailing step windows; and
* objective/gradient audits at exact, initial, selected, and final parameters.

Every result produced here is diagnostic.  LayerNorm runs are non-G1; a
LayerNorm-free architecture is only G1-compatible when its unperturbed
constructed weights are loaded exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from bf_transformer import Config
from bf_transformer.model import BFTransformer
from train.v3_diagnostic import (
    DiagnosticRecipe,
    _batches,
    _parameter_distance,
    collate_diagnostic,
    diagnostic_loss,
    evaluate,
    generate_evaluation_set,
)
from train.v3_model import V3BFTransformer


EXACTNESS_METRICS = (
    "graph_exact",
    "node_exact",
    "reachable_non_source_exact",
)


@dataclass(frozen=True)
class SaturationCriterion:
    """Task-metric plateau rule fixed before the v4 saturation results."""

    window_steps: int = 10_000
    maximum_best_gain: float = 0.01
    maximum_absolute_slope_per_window: float = 0.01
    maximum_window_range: float = 0.02


def _window_slope(
    events: list[dict],
    metric: str,
    *,
    window_steps: int,
) -> float:
    """Least-squares metric change per complete criterion window."""
    x_values = [
        (float(event["step"]) - float(events[0]["step"])) / window_steps
        for event in events
    ]
    y_values = [float(event["validation"][metric]) for event in events]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator == 0.0:
        return math.nan
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator


def saturation_summary(
    history: list[dict],
    *,
    total_steps: int,
    minimum_lr_ratio: float,
    criterion: SaturationCriterion = SaturationCriterion(),
) -> dict:
    """Evaluate the fixed two-window task-metric saturation criterion."""
    if minimum_lr_ratio != 0.0:
        return {
            "saturated": False,
            "reason": "terminal learning rate is non-zero",
            "criterion": asdict(criterion),
        }
    if total_steps < 2 * criterion.window_steps:
        return {
            "saturated": False,
            "reason": "run is shorter than two saturation windows",
            "criterion": asdict(criterion),
        }
    previous_start = total_steps - 2 * criterion.window_steps
    final_start = total_steps - criterion.window_steps
    previous = [
        event
        for event in history
        if previous_start <= int(event["step"]) < final_start
    ]
    final = [
        event
        for event in history
        if final_start <= int(event["step"]) <= total_steps
    ]
    if len(previous) < 3 or len(final) < 3:
        return {
            "saturated": False,
            "reason": "fewer than three evaluations in a saturation window",
            "criterion": asdict(criterion),
            "previous_event_count": len(previous),
            "final_event_count": len(final),
        }

    metrics: dict[str, dict[str, float | bool]] = {}
    for metric in EXACTNESS_METRICS:
        previous_values = [
            float(event["validation"][metric]) for event in previous
        ]
        final_values = [
            float(event["validation"][metric]) for event in final
        ]
        best_gain = max(final_values) - max(previous_values)
        slope = _window_slope(
            final,
            metric,
            window_steps=criterion.window_steps,
        )
        value_range = max(final_values) - min(final_values)
        plateau = (
            best_gain < criterion.maximum_best_gain
            and abs(slope) < criterion.maximum_absolute_slope_per_window
            and value_range < criterion.maximum_window_range
        )
        metrics[metric] = {
            "previous_best": max(previous_values),
            "final_best": max(final_values),
            "best_gain": best_gain,
            "final_slope_per_window": slope,
            "final_range": value_range,
            "plateau": plateau,
        }
    return {
        "saturated": all(
            bool(metric_summary["plateau"])
            for metric_summary in metrics.values()
        ),
        "criterion": asdict(criterion),
        "previous_window": [previous_start, final_start],
        "final_window": [final_start, total_steps],
        "previous_event_count": len(previous),
        "final_event_count": len(final),
        "metrics": metrics,
        "objective_windows": {
            "previous_mean_validation_loss": sum(
                float(event["validation"]["supervised_loss"])
                for event in previous
            )
            / len(previous),
            "final_mean_validation_loss": sum(
                float(event["validation"]["supervised_loss"])
                for event in final
            )
            / len(final),
        },
    }


def objective_gradient_audit(
    model: V3BFTransformer,
    examples,
    *,
    cfg: Config,
    supervision: str,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """Measure full-set mean objective and its parameter gradient."""
    model.eval()
    model.zero_grad(set_to_none=True)
    objective = 0.0
    example_count = len(examples)
    for group in _batches(examples, batch_size):
        batch = collate_diagnostic(group, cfg).to(device)
        states = model.forward_with_states(batch.streams)
        loss = diagnostic_loss(
            states,
            batch,
            supervision=supervision,
            cfg=cfg,
        )
        weight = len(group) / example_count
        (loss * weight).backward()
        objective += loss.item() * weight
    squared_norm = 0.0
    maximum_absolute = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        squared_norm += parameter.grad.detach().pow(2).sum().item()
        maximum_absolute = max(
            maximum_absolute,
            parameter.grad.detach().abs().max().item(),
        )
    result = {
        "objective": objective,
        "gradient_l2": math.sqrt(squared_norm),
        "gradient_max_abs": maximum_absolute,
    }
    model.zero_grad(set_to_none=True)
    return result


def _model_for_recipe(
    recipe: DiagnosticRecipe,
    *,
    device: torch.device,
) -> V3BFTransformer:
    cfg = Config(n_max=recipe.n_max)
    return V3BFTransformer(
        cfg,
        tied=True,
        width=recipe.width,
        layer_norm=recipe.layer_norm,
        random_initialisation=False,
    ).to(device)


def _load_checkpoint(
    recipe: DiagnosticRecipe,
    path: Path,
    *,
    device: torch.device,
) -> V3BFTransformer:
    model = _model_for_recipe(recipe, device=device)
    model.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    return model


def _checkpoint_audit(
    model: V3BFTransformer,
    examples,
    *,
    cfg: Config,
    supervision: str,
    device: torch.device,
    constructed_reference: V3BFTransformer | None,
) -> dict:
    result = {
        "metrics": evaluate(
            model,
            examples,
            cfg=cfg,
            supervision=supervision,
            device=device,
        ),
        "objective_gradient": objective_gradient_audit(
            model,
            examples,
            cfg=cfg,
            supervision=supervision,
            device=device,
        ),
    }
    if constructed_reference is not None:
        result["distance_from_constructed"] = _parameter_distance(
            model,
            constructed_reference,
        )
    return result


def analyse_run(
    run_dir: Path,
    *,
    device_name: str,
    criterion: SaturationCriterion = SaturationCriterion(),
) -> dict:
    """Add v4 saturation and objective audits to one v3-format run."""
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    recipe = DiagnosticRecipe(**run["recipe"])
    cfg = Config(n_max=recipe.n_max)
    device = torch.device(device_name)
    validation = generate_evaluation_set(
        cfg,
        train_n_max=recipe.train_n_max,
        seed=recipe.data_seed + 1,
        count=512,
    )

    constructed_reference: V3BFTransformer | None = None
    exact_constructed: dict | None = None
    if not recipe.layer_norm:
        constructed_reference = _model_for_recipe(recipe, device=device)
        constructed_reference.load_constructed_weights(BFTransformer(cfg))
        exact_constructed = _checkpoint_audit(
            constructed_reference,
            validation,
            cfg=cfg,
            supervision=recipe.supervision,
            device=device,
            constructed_reference=constructed_reference,
        )

    checkpoints = {}
    for name in ("initial", "best", "final"):
        model = _load_checkpoint(
            recipe,
            run_dir / f"{name}.pt",
            device=device,
        )
        checkpoints[name] = _checkpoint_audit(
            model,
            validation,
            cfg=cfg,
            supervision=recipe.supervision,
            device=device,
            constructed_reference=constructed_reference,
        )

    result = {
        "phase": "v4_diagnostic_analysis",
        "source_run": str(run_dir),
        "g1_status": run["g1_status"],
        "recipe": run["recipe"],
        "saturation": saturation_summary(
            run["history"],
            total_steps=recipe.steps,
            minimum_lr_ratio=recipe.minimum_lr_ratio,
            criterion=criterion,
        ),
        "checkpoints": checkpoints,
        "exact_constructed": exact_constructed,
    }
    output_path = run_dir / "v4-analysis.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "V4_ANALYSIS_COMPLETE",
                "output": str(output_path),
                "g1_status": result["g1_status"],
                "saturated": result["saturation"]["saturated"],
                "exact_constructed": exact_constructed,
                "checkpoints": checkpoints,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-steps", type=int, default=10_000)
    args = parser.parse_args()
    analyse_run(
        args.run,
        device_name=args.device,
        criterion=SaturationCriterion(window_steps=args.window_steps),
    )


if __name__ == "__main__":
    main()
