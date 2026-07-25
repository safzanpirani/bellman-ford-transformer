"""M3 + M4: routing heads are exactly one-hot; one block == one relaxation round."""
import random

import torch

from bf_transformer import BFTransformer, Graph, encode_graph, DEFAULT_CONFIG
from bf_transformer.registers import RegisterMap
from bf_transformer.blocks.attention import build_edge_reads_source, build_node_gathers
from bf_transformer.blocks.round import build_cand_mlp, build_cand_clamp
from bf_transformer.verify import random_graph, verify_graph, reference_states

cfg = DEFAULT_CONFIG
reg = RegisterMap(cfg)


def test_edge_reads_source_one_hot_and_exact():
    g = Graph(n=4, edges=((0, 1, 3.0), (2, 3, 5.0), (1, 1, 0.0)))
    stream = encode_graph(g, source=0)
    head = build_edge_reads_source(reg, cfg)
    att = head.attention_pattern(stream)
    edge_base = 1 + cfg.n_max
    dists = {0: 0.0, 1: cfg.inf_b, 2: cfg.inf_b, 3: cfg.inf_b}
    for j, (u, v, w) in enumerate(g.edges):
        row = att[edge_base + j]
        assert row[1 + u].item() == 1.0           # exactly one-hot on source node
        off = row.clone(); off[1 + u] = 0.0
        assert off.max().item() < 1e-6            # M3 leakage bound (it is exactly 0)
    out = head(stream)
    for j, (u, v, w) in enumerate(g.edges):
        assert out[edge_base + j, reg.idx("SCRATCH_SRC_DIST")].item() == dists[u]
    # sink and real node tokens read the sink: nothing written. (Padding rows
    # have zero queries and do pick up uniform-attention junk, but they are
    # never attended to and never decoded.)
    assert torch.all(out[:1 + g.n, reg.idx("SCRATCH_SRC_DIST")] == 0)


def test_cand_is_dist_plus_w_exact():
    g = Graph(n=3, edges=((0, 1, 7.0), (1, 2, 2.0)))
    stream = encode_graph(g, source=0)
    x = build_edge_reads_source(reg, cfg)(stream)
    x = build_cand_mlp(reg, cfg)(x)
    x = build_cand_clamp(reg, cfg)(x)
    edge_base = 1 + cfg.n_max
    assert x[edge_base + 0, reg.idx("CAND")].item() == 7.0          # dist[0]=0 + 7
    assert x[edge_base + 1, reg.idx("CAND")].item() == cfg.inf_b    # INF source clamps to INF_B
    assert torch.all(x[..., reg.idx("SCRATCH_SRC_DIST")] == 0)      # scratch cleared


def test_gather_heads_route_by_slot():
    g = Graph(n=3, edges=((0, 2, 4.0), (1, 2, 6.0)))
    stream = encode_graph(g, source=0)
    x = build_edge_reads_source(reg, cfg)(stream)
    x = build_cand_mlp(reg, cfg)(x)
    x = build_cand_clamp(reg, cfg)(x)
    gather = build_node_gathers(reg, cfg)
    y = gather(x)
    node2 = 1 + 2
    assert y[node2, reg.idx_in("SCRATCH_GATHER", 0)].item() == 4.0       # slot 0 edge
    assert y[node2, reg.idx_in("SCRATCH_GATHER", 1)].item() == cfg.inf_b  # slot 1 edge (INF src, clamped)
    # empty slots fall back to self: dist[node 2] = INF_B
    for k in range(2, cfg.d_max):
        assert y[node2, reg.idx_in("SCRATCH_GATHER", k)].item() == cfg.inf_b
    # node 0 has no in-edges: every head returns its own dist (0.0)
    assert torch.all(y[1 + 0, reg.slice("SCRATCH_GATHER")] == 0.0)
    # one-hot check across all heads/queries on real tokens
    real_rows = list(range(1 + g.n)) + [1 + cfg.n_max + j for j in range(len(g.edges))]
    for k in range(cfg.d_max):
        att = gather.attention_pattern(x, head=k)
        real = att[real_rows]
        assert torch.all((real.max(dim=-1).values - 1.0).abs() < 1e-6)


def test_one_block_equals_one_reference_round():
    torch.manual_seed(0)
    rng = random.Random(42)
    model = BFTransformer()
    from bf_transformer.encode import decode_dist
    import math
    for _ in range(100):
        g, s = random_graph(rng, cfg)
        want = reference_states(g, s, 1)[1]
        got = decode_dist(model.apply_round(encode_graph(g, s)), g.n)
        for i in range(g.n):
            if math.isinf(want[i]):
                assert math.isinf(got[i])
            else:
                assert got[i] == want[i]


def test_converged_state_is_fixed_point():
    """Delta-write-back: running the block on a converged state changes nothing."""
    rng = random.Random(7)
    model = BFTransformer()
    for _ in range(20):
        g, s = random_graph(rng, cfg)
        stream = model(encode_graph(g, s))          # converged after R rounds
        again = model.apply_round(stream.clone())
        assert torch.equal(stream, again)
