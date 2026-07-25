"""M5: full model vs networkx, FR-V executable induction, frozen parameters."""
import math
import random

import networkx as nx
import pytest
import torch
from torch import nn

from bf_transformer import (BFTransformer, Graph, encode_graph, decode_dist,
                            DEFAULT_CONFIG)
from bf_transformer.verify import random_graph, verify_graph

cfg = DEFAULT_CONFIG


def nx_distances(g: Graph, s: int) -> dict[int, float]:
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(g.n))
    for (u, v, w) in g.edges:
        G.add_edge(u, v, weight=w)
    lengths = nx.single_source_bellman_ford_path_length(G, s)
    return {i: lengths.get(i, math.inf) for i in range(g.n)}


def assert_matches_nx(model, g, s, atol=0.0):
    want = nx_distances(g, s)
    got = decode_dist(model(encode_graph(g, s)), g.n)
    for i in range(g.n):
        if math.isinf(want[i]):
            assert math.isinf(got[i]), (g, s, i, got[i])
        else:
            assert abs(got[i] - want[i]) <= atol, (g, s, i, want[i], got[i])


def test_full_model_random_graphs_integer_exact():
    rng = random.Random(2024)
    model = BFTransformer()
    for _ in range(100):
        g, s = random_graph(rng, cfg)
        assert_matches_nx(model, g, s, atol=0.0)


def test_full_model_float_weights():
    rng = random.Random(99)
    model = BFTransformer()
    for _ in range(100):
        g, s = random_graph(rng, cfg, integer_weights=False)
        assert_matches_nx(model, g, s, atol=1e-4)


@pytest.mark.parametrize("g,s", [
    (Graph(n=2, edges=()), 0),                                        # source, no out-edges
    (Graph(n=4, edges=((0, 1, 0.0), (1, 2, 0.0))), 0),                # zero weights + disconnected
    (Graph(n=3, edges=((0, 0, 5.0), (0, 1, cfg.w_max))), 0),          # self-loop + max weight
    (Graph(n=3, edges=((0, 1, 9.0), (0, 1, 2.0), (1, 2, 1.0))), 0),   # parallel edges
    (Graph(n=2, edges=((1, 0, 1.0),)), 0),                            # wrong-direction edge
])
def test_special_cases(g, s):
    assert_matches_nx(BFTransformer(), g, s)


def test_complete_graph_at_d_max():
    n = cfg.n_max
    rng = random.Random(5)
    edges = tuple((u, v, float(rng.randint(0, int(cfg.w_max))))
                  for v in range(n) for u in range(n) if u != v)
    g = Graph(n=n, edges=edges)
    assert_matches_nx(BFTransformer(), g, 0)


def test_fr_v_executable_induction_100_graphs():
    rng = random.Random(1234)
    model = BFTransformer()
    for _ in range(100):
        g, s = random_graph(rng, cfg)
        verify_graph(model, g, s)


def test_zero_trained_parameters():
    model = BFTransformer()
    n_params = 0
    for p in model.parameters():
        n_params += 1
        assert p.requires_grad is False
    assert n_params > 0
    # every weight-bearing module is tagged by its construction function
    for m in model.modules():
        if any(True for _ in m.parameters(recurse=False)):
            assert getattr(m, "constructed", False) or isinstance(m, nn.ParameterList)


def test_no_layernorm_no_dropout():
    for m in BFTransformer().modules():
        assert not isinstance(m, (nn.LayerNorm, nn.Dropout, nn.BatchNorm1d))
