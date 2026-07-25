"""(v2) Soft-min reducer: attention-based min instead of the ReLU tournament.

One head. Node v's candidate set is {v itself} ∪ {in-edges of v}; every member
gets a large membership bonus (4C) plus a value-dependent logit -c_soft * val
(val = DIST for the self token, CAND for edge tokens). Within the candidate
set the softmax therefore computes weights ∝ exp(-c_soft * val): a soft-min.
Non-members sit >= ~2C lower and contribute exactly 0 (float32 underflow), so
the *routing* stays exact; only the reduction is soft.

Output goes straight to SCRATCH_MIN; the standard write-back layer then adds
-ReLU(DIST - SCRATCH_MIN), which also keeps distances monotonically
non-increasing under the soft reducer.
"""
import torch

from ..config import Config
from ..registers import RegisterMap
from .attention import HandAttention


def build_soft_gather(reg: RegisterMap, cfg: Config, c_soft: float) -> HandAttention:
    n = cfg.n_max
    c = cfg.c_match
    # membership bonus must dominate the value term by a float32-underflow gap
    if c_soft * (cfg.inf_b + cfg.w_max) > 2.0 * c - 250.0:
        raise ValueError(f"c_soft={c_soft} too large for c_hard={cfg.c_hard}: "
                         "routing would leak")
    dk = n + 1
    wq = torch.zeros(reg.d_model, dk)
    wk = torch.zeros(reg.d_model, dk)
    wv = torch.zeros(reg.d_model, 1)
    wo = torch.zeros(1, reg.d_model)

    et, ni = reg.slice("EDGE_TO"), reg.slice("NODE_ID")
    for j in range(n):
        wq[ni.start + j, j] = 1.0
        wk[et.start + j, j] = 4.0 * c   # in-edge membership
        wk[ni.start + j, j] = 4.0 * c   # self membership
    wq[reg.idx("ONES"), n] = 1.0
    wk[reg.idx("SINK"), n] = 1.0 * c    # fallback target for edge/sink queries
    # value-dependent logits (the soft-min temperature)
    wk[reg.idx("DIST"), n] = -c_soft
    wk[reg.idx("CAND"), n] = -c_soft

    wv[reg.idx("DIST"), 0] = 1.0
    wv[reg.idx("CAND"), 0] = 1.0
    wo[0, reg.idx("SCRATCH_MIN")] = 1.0
    return HandAttention([(wq, wk, wv, wo)])
