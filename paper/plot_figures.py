"""Generate publication figures from the frozen v4 curve artifact."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "docs" / "pilot-v4-curves.json"
OUTPUT = ROOT / "paper" / "figures" / "saturation-curves.pdf"


def series(curves: dict, run: str, metric: str) -> tuple[list[int], list[float]]:
    history = curves[run]["history"]
    steps = [int(event["step"]) for event in history]
    values = [100.0 * float(event["validation"][metric]) for event in history]
    return steps, values


def main() -> None:
    payload = json.loads(CURVES.read_text())
    curves = payload["curves"]
    output_parent = OUTPUT.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.2,
            "axes.labelsize": 8.2,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.05), sharex=True, sharey=True)
    runs = (
        ("t1_ln_width128_80k_lr3e4", "Pre-LN (non-G1)", "#2f6f9f", "-"),
        ("t2_noln_width128_80k_lr3e4", "No LN (G1)", "#b44b3c", "--"),
    )
    panels = (
        ("graph_exact", "Graph exactness"),
        ("reachable_non_source_exact", "Reachable non-source exactness"),
    )

    for axis, (metric, title) in zip(axes, panels, strict=True):
        for run, label, color, linestyle in runs:
            steps, values = series(curves, run, metric)
            axis.plot(
                [step / 1000 for step in steps],
                values,
                color=color,
                linewidth=1.45,
                linestyle=linestyle,
                label=label,
            )
        axis.set_title(title, pad=4)
        axis.set_xlabel("Optimization step (thousands)")
        axis.set_xlim(0, 80)
        axis.set_ylim(0, 101)
        axis.xaxis.set_major_locator(MultipleLocator(10))
        axis.yaxis.set_major_locator(MultipleLocator(20))
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        axis.grid(color="#d9d9d9", linewidth=0.45, alpha=0.85)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Validation exactness")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout(w_pad=2.0, pad=0.25)
    fig.savefig(OUTPUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
