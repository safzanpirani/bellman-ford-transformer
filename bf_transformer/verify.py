"""FR-V: the inductive proof, made executable.

A pure-Python synchronous (Jacobi) Bellman-Ford serves as the round-by-round
reference. ``verify_graph`` asserts that after k applications of the block,
the decoded DIST register equals the reference state after k relaxation
rounds, for every k. Run as a script to check 100 random graphs.
"""
import math
import random

import torch

from .config import Config, DEFAULT_CONFIG
from .encode import Graph, encode_graph, decode_dist
from .model import BFTransformer


def reference_round(dist: list[float], g: Graph) -> list[float]:
    """One synchronous relaxation round (matches the block's semantics:
    every node minimises over candidates computed from the *previous* state)."""
    new = list(dist)
    for (u, v, w) in g.edges:
        if dist[u] < math.inf:
            new[v] = min(new[v], dist[u] + w)
    return new


def reference_states(g: Graph, source: int, rounds: int) -> list[list[float]]:
    """States after 0, 1, ..., rounds synchronous rounds."""
    dist = [math.inf] * g.n
    dist[source] = 0.0
    states = [list(dist)]
    for _ in range(rounds):
        dist = reference_round(dist, g)
        states.append(list(dist))
    return states


def verify_graph(model: BFTransformer, g: Graph, source: int,
                 atol: float = 1e-4) -> None:
    """Assert round-by-round equality of model state and reference state."""
    cfg = model.cfg
    states = reference_states(g, source, cfg.rounds)
    stream = encode_graph(g, source, cfg, model.reg)
    for k in range(cfg.rounds + 1):
        got = decode_dist(stream, g.n, cfg, model.reg)
        for i in range(g.n):
            want = states[k][i]
            if math.isinf(want):
                assert math.isinf(got[i]), \
                    f"round {k}, node {i}: expected inf, got {got[i]} (graph={g}, s={source})"
            else:
                assert abs(got[i] - want) <= atol, \
                    f"round {k}, node {i}: expected {want}, got {got[i]} (graph={g}, s={source})"
        if k < cfg.rounds:
            stream = model.apply_round(stream)


def random_graph(rng: random.Random, cfg: Config = DEFAULT_CONFIG,
                 n: int | None = None, integer_weights: bool = True,
                 allow_negative: bool = False) -> tuple[Graph, int]:
    """Random graph respecting N_MAX / W_MAX / D_MAX. Naturally produces
    self-loops, parallel edges, isolated nodes and zero-weight edges."""
    n = n if n is not None else rng.randint(2, cfg.n_max)
    lo = -int(cfg.w_max) if allow_negative else 0
    edges = []
    for v in range(n):
        for _ in range(rng.randint(0, cfg.d_max)):
            u = rng.randrange(n)
            if integer_weights:
                w = float(rng.randint(lo, int(cfg.w_max)))
            else:
                w = rng.uniform(lo, cfg.w_max)
            edges.append((u, v, w))
    rng.shuffle(edges)
    return Graph(n=n, edges=tuple(edges)), rng.randrange(n)


def main() -> None:
    torch.manual_seed(0)
    rng = random.Random(1234)
    model = BFTransformer()
    for i in range(100):
        g, s = random_graph(rng)
        verify_graph(model, g, s)
    print("FR-V: 100/100 random graphs verified round-by-round.")


if __name__ == "__main__":
    main()
