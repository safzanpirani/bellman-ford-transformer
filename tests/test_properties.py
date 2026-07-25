"""FR6: the 1,000-graph property suite (seeded batch + hypothesis)."""
import math
import random

import torch
from hypothesis import given, settings, strategies as st

from bf_transformer import BFTransformer, Graph, encode_graph, decode_dist, DEFAULT_CONFIG
from bf_transformer.verify import random_graph
from tests.test_full_model import nx_distances

cfg = DEFAULT_CONFIG
model = BFTransformer()


def test_1000_random_graphs_batched():
    """FR6: >= 1000 random graphs across sizes 2..N_MAX, integer weights, exact."""
    rng = random.Random(31337)
    graphs = []
    for i in range(1000):
        # sweep sizes deterministically so every size 2..n_max is well covered
        n = 2 + (i % (cfg.n_max - 1))
        graphs.append(random_graph(rng, cfg, n=n))
    streams = torch.stack([encode_graph(g, s) for g, s in graphs])
    out = model(streams)  # one batched forward for all 1000 graphs
    for row, (g, s) in zip(out, graphs):
        want = nx_distances(g, s)
        got = decode_dist(row, g.n)
        for i in range(g.n):
            if math.isinf(want[i]):
                assert math.isinf(got[i]), (g, s, i)
            else:
                assert got[i] == want[i], (g, s, i, want[i], got[i])


@st.composite
def graphs(draw):
    n = draw(st.integers(min_value=2, max_value=cfg.n_max))
    edges = []
    for v in range(n):
        indeg = draw(st.integers(min_value=0, max_value=cfg.d_max))
        for _ in range(indeg):
            u = draw(st.integers(min_value=0, max_value=n - 1))
            w = draw(st.integers(min_value=0, max_value=int(cfg.w_max)))
            edges.append((u, v, float(w)))
    s = draw(st.integers(min_value=0, max_value=n - 1))
    return Graph(n=n, edges=tuple(edges)), s


@settings(max_examples=200, deadline=None, derandomize=True)
@given(graphs())
def test_hypothesis_matches_networkx(gs):
    g, s = gs
    want = nx_distances(g, s)
    got = decode_dist(model(encode_graph(g, s)), g.n)
    for i in range(g.n):
        if math.isinf(want[i]):
            assert math.isinf(got[i])
        else:
            assert got[i] == want[i]
