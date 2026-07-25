"""Graph -> initial residual stream, and residual stream -> distances.

Token layout (fixed, padded):
    row 0                          sink token (all-zero payload, SINK=1)
    rows 1 .. n                    node tokens (zero-padded up to n_max)
    rows 1+n_max .. 1+n_max+|E|-1  edge tokens (zero-padded up to e_max)

Padding rows are all-zero (ONES=0). They are never attended to (zero keys
lose every logit competition) and never decoded.
"""
import math
from dataclasses import dataclass

import torch

from .config import Config, DEFAULT_CONFIG
from .registers import RegisterMap


@dataclass(frozen=True)
class Graph:
    """Directed graph: nodes 0..n-1, edges are (u, v, w) triples."""
    n: int
    edges: tuple[tuple[int, int, float], ...]


def validate_graph(g: Graph, source: int, cfg: Config = DEFAULT_CONFIG) -> None:
    if not (1 <= g.n <= cfg.n_max):
        raise ValueError(f"graph has {g.n} nodes; supported range is 1..{cfg.n_max}")
    if not (0 <= source < g.n):
        raise ValueError(f"source {source} not a node of the graph")
    if len(g.edges) > cfg.e_max:
        raise ValueError(f"graph has {len(g.edges)} edges; max is {cfg.e_max}")
    indeg = [0] * g.n
    for (u, v, w) in g.edges:
        if not (0 <= u < g.n and 0 <= v < g.n):
            raise ValueError(f"edge ({u},{v}) references a non-existent node")
        if not math.isfinite(w) or abs(w) > cfg.w_max:
            raise ValueError(f"edge weight {w} outside [-{cfg.w_max}, {cfg.w_max}]")
        indeg[v] += 1
        if indeg[v] > cfg.d_max:
            raise ValueError(f"node {v} has in-degree > D_MAX={cfg.d_max}; "
                             "graph rejected (never silently truncated)")


def encode_graph(g: Graph, source: int, cfg: Config = DEFAULT_CONFIG,
                 reg: RegisterMap | None = None) -> torch.Tensor:
    """Build the initial residual stream [seq_len, d_model] for (G, s)."""
    validate_graph(g, source, cfg)
    reg = reg or RegisterMap(cfg)
    stream = torch.zeros(cfg.seq_len, reg.d_model, dtype=torch.float32)

    # sink token
    stream[0, reg.idx("ONES")] = 1.0
    stream[0, reg.idx("SINK")] = 1.0

    # node tokens
    for i in range(g.n):
        row = 1 + i
        stream[row, reg.idx("ONES")] = 1.0
        stream[row, reg.idx("IS_NODE")] = 1.0
        stream[row, reg.idx_in("NODE_ID", i)] = 1.0
        stream[row, reg.idx("DIST")] = 0.0 if i == source else cfg.inf_b

    # edge tokens; slots are assigned per *destination* so that the D_MAX
    # gather heads each see at most one matching edge per node.
    next_slot = [0] * g.n
    edge_base = 1 + cfg.n_max
    for j, (u, v, w) in enumerate(g.edges):
        row = edge_base + j
        stream[row, reg.idx("ONES")] = 1.0
        stream[row, reg.idx("IS_EDGE")] = 1.0
        stream[row, reg.idx_in("EDGE_FROM", u)] = 1.0
        stream[row, reg.idx_in("EDGE_TO", v)] = 1.0
        stream[row, reg.idx("EDGE_W")] = float(w)
        stream[row, reg.idx_in("SLOT", next_slot[v])] = 1.0
        next_slot[v] += 1

    return stream


def decode_dist(stream: torch.Tensor, n: int, cfg: Config = DEFAULT_CONFIG,
                reg: RegisterMap | None = None) -> dict[int, float]:
    """Read DIST off the node tokens; values >= INF_B - 0.5 decode as inf."""
    reg = reg or RegisterMap(cfg)
    out: dict[int, float] = {}
    node_rows = stream[..., 1:1 + n, reg.idx("DIST")]
    for i in range(n):
        v = float(node_rows[..., i])
        out[i] = math.inf if v >= cfg.inf_b - 0.5 else v
    return out


def decode_negative_cycle_flag(stream: torch.Tensor, n: int,
                               cfg: Config = DEFAULT_CONFIG,
                               reg: RegisterMap | None = None,
                               threshold: float = 1e-3) -> bool:
    """True iff any node's DIST decreased during the extra detection round."""
    reg = reg or RegisterMap(cfg)
    flags = stream[..., 1:1 + n, reg.idx("FLAG_DEC")]
    return bool(flags.max().item() > threshold)
