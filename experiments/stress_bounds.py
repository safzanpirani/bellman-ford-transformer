"""Stress test: where does the hand-coded construction stop being exact?

Bypasses Config.__post_init__ guards to sweep (a) the attention logit scale
c_hard below the declared c_hard/4 > 200 bound and (b) the weight magnitude
w_max past the float32 comfort zone. For each setting, reports the fraction
of random graphs whose final DIST matches the reference exactly, and the max
absolute error.

Usage: python -m experiments.stress_bounds [--graphs 40] [--seed 0]
"""
import argparse
import dataclasses
import math
import random

import torch

from bf_transformer import BFTransformer, encode_graph
from bf_transformer.config import Config
from bf_transformer.registers import RegisterMap
from bf_transformer.verify import random_graph, reference_states


def unchecked_config(**kw) -> Config:
    """Config with the validation guards skipped (deliberately out of spec)."""
    cfg = object.__new__(Config)
    for f in dataclasses.fields(Config):
        object.__setattr__(cfg, f.name, kw.get(f.name, f.default))
    return cfg


def score(cfg: Config, n_graphs: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    reg = RegisterMap(cfg)
    model = BFTransformer(cfg)
    graphs = [random_graph(rng, cfg) for _ in range(n_graphs)]
    with torch.no_grad():
        out = model(torch.stack([encode_graph(g, s, cfg, reg) for g, s in graphs]))
    exact, worst = 0, 0.0
    for row, (g, s) in zip(out, graphs):
        truth = reference_states(g, s, cfg.rounds)[-1]
        got = row[1:1 + g.n, reg.idx("DIST")].tolist()
        errs = [abs(min(t, cfg.inf_b) - x) if math.isfinite(x) else math.inf
                for t, x in zip(truth, got)]
        e = max(errs)
        worst = max(worst, e)
        exact += e == 0.0
    return exact / n_graphs, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("== logit scale sweep (w_max=10, guard requires c_hard/4 > 200) ==")
    print(f"{'c_hard':>8} {'c_match':>8} {'exact%':>7} {'max_err':>10}")
    for c in [8, 16, 32, 64, 128, 256, 400, 420, 480, 800, 1e4]:
        frac, err = score(unchecked_config(c_hard=float(c)), a.graphs, a.seed)
        print(f"{c:>8g} {c / 4:>8g} {100 * frac:>6.1f}% {err:>10.4g}")

    print("\n== weight magnitude sweep (c_hard=1e4, guard requires inf_b+w_max < 1e6) ==")
    print(f"{'w_max':>10} {'inf_b':>10} {'exact%':>7} {'max_err':>10}")
    for w in [10, 1e3, 1e4, 1e5, 2e5, 1e6, 2e6, 4e6, 1e7]:
        frac, err = score(unchecked_config(w_max=float(w)), a.graphs, a.seed)
        print(f"{w:>10g} {8 * w + 1:>10g} {100 * frac:>6.1f}% {err:>10.4g}")


if __name__ == "__main__":
    main()
