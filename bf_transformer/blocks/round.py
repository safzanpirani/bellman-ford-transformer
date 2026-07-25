"""One Bellman-Ford round as a sequence of residual sublayers.

Synchronous (Jacobi) relaxation:
    new_dist[v] = min( dist[v], min over in-edges (u,v,w) of dist[u] + w )

Sublayer plan (exact reducer):
  1. attention: edge tokens read dist[u]  ->  SCRATCH_SRC_DIST
  2. MLP: CAND = SCRATCH_SRC_DIST + EDGE_W, with INF guard (if dist[u] is
     INF_B the candidate is pushed above INF_B); clears SCRATCH_SRC_DIST
  3. MLP: clamp CAND down to INF_B  (so INF never grows round over round)
  4. attention: D_MAX gather heads  ->  SCRATCH_GATHER[0..D)
  5. MLP: SCRATCH_MIN = DIST
  6. D_MAX MLPs: SCRATCH_MIN <- min(SCRATCH_MIN, SCRATCH_GATHER[k])
  7. MLP: DIST += -ReLU(DIST - SCRATCH_MIN)   (delta write-back),
     and clear all scratch registers so the weight-tied block can run again.

Every sublayer adds a delta to the residual stream; nothing is overwritten.
"""
from torch import nn

from ..config import Config
from ..registers import RegisterMap
from .attention import build_edge_reads_source, build_node_gathers
from .minred import MLPBuilder
from .soft import build_soft_gather


def build_cand_mlp(reg: RegisterMap, cfg: Config):
    """CAND += (src + w), plus the INF guard; SCRATCH_SRC_DIST cleared."""
    src, w, cand = reg.idx("SCRATCH_SRC_DIST"), reg.idx("EDGE_W"), reg.idx("CAND")
    b = MLPBuilder(reg.d_model)
    # candidate value (exact for either sign)
    b.unit({src: 1.0, w: 1.0}, 0.0, {cand: 1.0})
    b.unit({src: -1.0, w: -1.0}, 0.0, {cand: -1.0})
    # INF guard: if dist[u] >= INF_B - 0.5, push the candidate up by
    # k_push * (dist[u] - (INF_B - 0.5)) >= 2*(w_max+1) > w_max, guaranteeing
    # cand > INF_B even for the most negative edge weight. Real distances are
    # always <= (n-1)*w_max < INF_B - 0.5 - w_max, so the guard never fires
    # on reachable sources.
    b.unit({src: 1.0}, -(cfg.inf_b - 0.5), {cand: cfg.k_push})
    # clear the scratch register for the next round
    b.clear(src)
    return b.build()


def build_cand_clamp(reg: RegisterMap, cfg: Config):
    """CAND += -ReLU(CAND - INF_B): unreachable candidates become exactly INF_B."""
    cand = reg.idx("CAND")
    return MLPBuilder(reg.d_model).unit({cand: 1.0}, -cfg.inf_b, {cand: -1.0}).build()


def build_min_init(reg: RegisterMap):
    """SCRATCH_MIN += DIST (exact copy; edge/sink tokens copy 0)."""
    return MLPBuilder(reg.d_model).copy(reg.idx("DIST"), reg.idx("SCRATCH_MIN")).build()


def build_min_step(reg: RegisterMap, k: int):
    """SCRATCH_MIN <- min(SCRATCH_MIN, SCRATCH_GATHER[k])."""
    b = MLPBuilder(reg.d_model)
    b.min_fold(reg.idx("SCRATCH_MIN"), reg.idx_in("SCRATCH_GATHER", k))
    return b.build()


def build_writeback(reg: RegisterMap, cfg: Config, clear_gather: bool = True):
    """DIST += -ReLU(DIST - SCRATCH_MIN); clear every scratch register.

    The write is the *delta* between the old distance and the round minimum —
    never an overwrite — so applying the block to a converged state is a
    fixed point.
    """
    b = MLPBuilder(reg.d_model)
    b.unit({reg.idx("DIST"): 1.0, reg.idx("SCRATCH_MIN"): -1.0}, 0.0,
           {reg.idx("DIST"): -1.0})
    b.clear(reg.idx("SCRATCH_MIN"))
    b.clear(reg.idx("CAND"))
    if clear_gather:
        for k in range(cfg.d_max):
            b.clear(reg.idx_in("SCRATCH_GATHER", k))
    return b.build()


class BFRound(nn.Module):
    """One relaxation round: ordered residual sublayers."""

    def __init__(self, layers: list[nn.Module], names: list[str] | None = None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_names = names or [f"layer{i}" for i in range(len(layers))]
        self.constructed = True

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def trace(self, x):
        """Yield (name, module, state_after) for each sublayer — used by the
        visualisation exporter and for debugging."""
        for name, layer in zip(self.layer_names, self.layers):
            x = layer(x)
            yield name, layer, x


def build_round(reg: RegisterMap, cfg: Config, reducer: str = "exact",
                c_soft: float = 1.0) -> BFRound:
    layers: list[nn.Module] = [
        build_edge_reads_source(reg, cfg),
        build_cand_mlp(reg, cfg),
        build_cand_clamp(reg, cfg),
    ]
    names = ["attn:edge_reads_source", "mlp:candidate", "mlp:clamp_inf"]
    if reducer == "exact":
        layers.append(build_node_gathers(reg, cfg))
        names.append("attn:node_gathers")
        layers.append(build_min_init(reg))
        names.append("mlp:min_init")
        for k in range(cfg.d_max):
            layers.append(build_min_step(reg, k))
            names.append(f"mlp:min_fold[{k}]")
        layers.append(build_writeback(reg, cfg, clear_gather=True))
        names.append("mlp:writeback")
    elif reducer == "soft":
        layers.append(build_soft_gather(reg, cfg, c_soft))  # -> SCRATCH_MIN directly
        names.append("attn:soft_gather")
        layers.append(build_writeback(reg, cfg, clear_gather=False))
        names.append("mlp:writeback")
    else:
        raise ValueError(f"unknown reducer {reducer!r}")
    return BFRound(layers, names)
