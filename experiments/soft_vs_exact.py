"""FR8: exact ReLU-tournament reducer vs attention soft-min reducer.

Sweeps the soft-min temperature C_SOFT and the round count, measuring the
max absolute distance error against ground truth (networkx) over a fixed
set of random graphs. The exact reducer is plotted as the zero-error
baseline.

Usage:
    python -m experiments.soft_vs_exact [--graphs 50] [--seed 0] [--out experiments/soft_vs_exact.png]
"""
import argparse
import math
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from bf_transformer import BFTransformer, encode_graph, DEFAULT_CONFIG
from bf_transformer.registers import RegisterMap
from bf_transformer.verify import random_graph, reference_states

C_SOFT_SWEEP = [0.5, 1.0, 2.0, 5.0, 10.0]


def max_error_per_round(model, graphs, cfg, max_rounds):
    """max over graphs/nodes of |decoded - true| after k rounds, for each k.

    Unreachable nodes are compared in clamped space (true INF == INF_B), so
    the metric stays finite and reflects what the model can actually express.
    """
    reg = RegisterMap(cfg)
    streams = torch.stack([encode_graph(g, s, cfg, reg) for g, s in graphs])
    truth = [reference_states(g, s, cfg.rounds)[-1] for g, s in graphs]
    errs = []
    for k in range(1, max_rounds + 1):
        streams = model.apply_round(streams)
        worst = 0.0
        for row, (g, s), tr in zip(streams, graphs, truth):
            dist = row[1:1 + g.n, reg.idx("DIST")]
            for i in range(g.n):
                want = min(tr[i], cfg.inf_b) if not math.isinf(tr[i]) else cfg.inf_b
                worst = max(worst, abs(dist[i].item() - want))
        errs.append(worst)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=None,
                    help="rounds to sweep (default: 2 * (n_max - 1))")
    ap.add_argument("--out", default="experiments/soft_vs_exact.png")
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    max_rounds = args.rounds or 2 * cfg.rounds
    rng = random.Random(args.seed)
    graphs = [random_graph(rng, cfg) for _ in range(args.graphs)]

    plt.figure(figsize=(8, 5))
    exact_errs = max_error_per_round(BFTransformer(cfg), graphs, cfg, max_rounds)
    plt.plot(range(1, max_rounds + 1), exact_errs, "k-o", label="exact (ReLU tournament)")
    print(f"exact reducer: max error after {max_rounds} rounds = {exact_errs[-1]:.3g}")

    for c_soft in C_SOFT_SWEEP:
        model = BFTransformer(cfg, reducer="soft", c_soft=c_soft)
        errs = max_error_per_round(model, graphs, cfg, max_rounds)
        plt.plot(range(1, max_rounds + 1), errs, "--s", label=f"soft, C_SOFT={c_soft}")
        print(f"soft C_SOFT={c_soft:>4}: max error per round = "
              + " ".join(f"{e:.3g}" for e in errs))

    plt.yscale("symlog", linthresh=1e-6)
    plt.xlabel("rounds")
    plt.ylabel("max |dist error| over all graphs/nodes")
    plt.title(f"Exact vs soft-min Bellman-Ford ({args.graphs} random graphs, "
              f"N_MAX={cfg.n_max})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"plot saved to {args.out}")


if __name__ == "__main__":
    main()
