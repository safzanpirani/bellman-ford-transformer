"""Weight-tied Bellman-Ford transformer.

One hand-constructed round block, applied R = N_MAX - 1 times (looped /
weight-tied presentation). No LayerNorm, no dropout, no positional
embeddings, no trained parameters anywhere.
"""
import torch
from torch import nn

from .config import Config, DEFAULT_CONFIG
from .registers import RegisterMap
from .blocks.minred import MLPBuilder
from .blocks.round import build_round


def build_snapshot_layer(reg: RegisterMap):
    """SCRATCH_PREV_DIST += DIST (taken just before the detection round)."""
    return MLPBuilder(reg.d_model).copy(reg.idx("DIST"), reg.idx("SCRATCH_PREV_DIST")).build()


def build_flag_layer(reg: RegisterMap):
    """FLAG_DEC += ReLU(prev - dist): hot iff this node's DIST decreased in
    the extra round (i.e. a reachable negative cycle exists). Clears the
    snapshot register."""
    b = MLPBuilder(reg.d_model)
    b.unit({reg.idx("SCRATCH_PREV_DIST"): 1.0, reg.idx("DIST"): -1.0}, 0.0,
           {reg.idx("FLAG_DEC"): 1.0})
    b.clear(reg.idx("SCRATCH_PREV_DIST"))
    return b.build()


class BFTransformer(nn.Module):
    def __init__(self, cfg: Config = DEFAULT_CONFIG, reducer: str = "exact",
                 c_soft: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.reg = RegisterMap(cfg)
        self.block = build_round(self.reg, cfg, reducer=reducer, c_soft=c_soft)
        self.snapshot = build_snapshot_layer(self.reg)
        self.flag = build_flag_layer(self.reg)
        for p in self.parameters():
            assert not p.requires_grad

    def apply_round(self, stream: torch.Tensor) -> torch.Tensor:
        return self.block(stream)

    def forward(self, stream: torch.Tensor, rounds: int | None = None) -> torch.Tensor:
        r = self.cfg.rounds if rounds is None else rounds
        for _ in range(r):
            stream = self.block(stream)
        return stream

    def forward_with_cycle_check(self, stream: torch.Tensor) -> torch.Tensor:
        """R rounds, snapshot DIST, one extra round, compare (FR7)."""
        stream = self.forward(stream)
        stream = self.snapshot(stream)
        stream = self.block(stream)
        stream = self.flag(stream)
        return stream
