"""M6 / FR7: negative weights and the negative-cycle flag."""
import math
import random

import networkx as nx

from bf_transformer import (BFTransformer, Graph, encode_graph, decode_dist,
                            decode_negative_cycle_flag, DEFAULT_CONFIG)
from bf_transformer.verify import random_graph

cfg = DEFAULT_CONFIG
model = BFTransformer()


def nx_has_reachable_negative_cycle(g: Graph, s: int):
    G = nx.MultiDiGraph()
    G.add_nodes_from(range(g.n))
    for (u, v, w) in g.edges:
        G.add_edge(u, v, weight=w)
    try:
        lengths = nx.single_source_bellman_ford_path_length(G, s)
        return False, {i: lengths.get(i, math.inf) for i in range(g.n)}
    except nx.NetworkXUnbounded:
        return True, None


def run(g, s):
    stream = model.forward_with_cycle_check(encode_graph(g, s))
    return decode_negative_cycle_flag(stream, g.n), decode_dist(stream, g.n)


def test_reachable_negative_cycle_flags():
    g = Graph(n=3, edges=((0, 1, 1.0), (1, 2, -5.0), (2, 1, 2.0)))
    flag, _ = run(g, 0)
    assert flag is True


def test_negative_edges_without_cycle_exact_and_unflagged():
    g = Graph(n=4, edges=((0, 1, 5.0), (1, 2, -3.0), (0, 2, 4.0), (2, 3, -1.0)))
    flag, dist = run(g, 0)
    assert flag is False
    assert dist == {0: 0.0, 1: 5.0, 2: 2.0, 3: 1.0}


def test_unreachable_negative_cycle_not_flagged():
    # cycle 2<->3 is negative but unreachable from source 0
    g = Graph(n=4, edges=((0, 1, 1.0), (2, 3, -4.0), (3, 2, 1.0)))
    flag, dist = run(g, 0)
    assert flag is False
    assert dist[1] == 1.0 and math.isinf(dist[2]) and math.isinf(dist[3])


def test_negative_self_loop():
    g = Graph(n=2, edges=((0, 1, 1.0), (1, 1, -1.0)))
    flag, _ = run(g, 0)
    assert flag is True


def test_random_negative_graphs_match_networkx():
    rng = random.Random(777)
    for _ in range(100):
        g, s = random_graph(rng, cfg, allow_negative=True)
        nx_flag, nx_dist = nx_has_reachable_negative_cycle(g, s)
        flag, dist = run(g, s)
        assert flag == nx_flag, (g, s)
        if not nx_flag:
            for i in range(g.n):
                if math.isinf(nx_dist[i]):
                    assert math.isinf(dist[i]), (g, s, i)
                else:
                    assert dist[i] == nx_dist[i], (g, s, i, nx_dist[i], dist[i])
