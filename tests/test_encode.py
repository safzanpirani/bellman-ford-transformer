"""M1: encoder/decoder round-trip and input validation."""
import math

import pytest
import torch

from bf_transformer import Graph, encode_graph, decode_dist, DEFAULT_CONFIG
from bf_transformer.registers import RegisterMap

cfg = DEFAULT_CONFIG
reg = RegisterMap(cfg)


def test_roundtrip_zero_rounds():
    g = Graph(n=4, edges=((0, 1, 3.0), (1, 2, 2.0)))
    stream = encode_graph(g, source=0)
    d = decode_dist(stream, g.n)
    assert d[0] == 0.0
    assert all(math.isinf(d[i]) for i in (1, 2, 3))


def test_inf_decode_threshold():
    g = Graph(n=2, edges=())
    stream = encode_graph(g, source=1)
    d = decode_dist(stream, g.n)
    assert math.isinf(d[0]) and d[1] == 0.0


def test_slots_unique_per_destination():
    g = Graph(n=3, edges=((0, 2, 1.0), (1, 2, 1.0), (2, 2, 1.0), (0, 1, 1.0)))
    stream = encode_graph(g, source=0)
    edge_rows = stream[1 + cfg.n_max:1 + cfg.n_max + len(g.edges), reg.slice("SLOT")]
    # edges into node 2 get slots 0,1,2; edge into node 1 gets slot 0
    assert edge_rows[0].argmax() == 0
    assert edge_rows[1].argmax() == 1
    assert edge_rows[2].argmax() == 2
    assert edge_rows[3].argmax() == 0


def test_padding_rows_are_zero():
    g = Graph(n=2, edges=((0, 1, 1.0),))
    stream = encode_graph(g, source=0)
    assert torch.all(stream[1 + g.n:1 + cfg.n_max] == 0)
    assert torch.all(stream[1 + cfg.n_max + 1:] == 0)


@pytest.mark.parametrize("g,s,msg", [
    (Graph(n=cfg.n_max + 1, edges=()), 0, "nodes"),
    (Graph(n=2, edges=((0, 3, 1.0),)), 0, "non-existent"),
    (Graph(n=2, edges=((0, 1, cfg.w_max + 1),)), 0, "weight"),
    (Graph(n=2, edges=tuple((0, 1, 1.0) for _ in range(cfg.d_max + 1))), 0, "in-degree"),
    (Graph(n=2, edges=()), 5, "source"),
])
def test_rejections(g, s, msg):
    with pytest.raises(ValueError, match=msg):
        encode_graph(g, s)
