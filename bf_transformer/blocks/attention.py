"""Hand-constructed attention heads.

Exact one-hot routing: every query has a unique maximum logit, and all
competing logits sit >= c_match (= C_HARD/4, > 200) below it. After the
softmax's internal max-subtraction, every off-target term is exp(-gap) with
gap > 200, which underflows float32 to exactly 0.0 — so attention weights are
*exactly* {1, 0, ...} and the value copy is float-exact.

Every token that has nothing to read attends to the sink token (all-zero
payload), so no junk is ever written into live registers.
"""
import torch
from torch import nn

from ..config import Config
from ..registers import RegisterMap


class HandAttention(nn.Module):
    """Multi-head attention with explicit per-head W_Q/W_K/W_V/W_O.

    No 1/sqrt(d_k) scaling (constants are folded into the keys), no masking,
    no output bias. Output is added to the residual stream.
    """

    def __init__(self, heads: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        super().__init__()
        self.wq = nn.ParameterList()
        self.wk = nn.ParameterList()
        self.wv = nn.ParameterList()
        self.wo = nn.ParameterList()
        for (wq, wk, wv, wo) in heads:
            self.wq.append(nn.Parameter(wq, requires_grad=False))
            self.wk.append(nn.Parameter(wk, requires_grad=False))
            self.wv.append(nn.Parameter(wv, requires_grad=False))
            self.wo.append(nn.Parameter(wo, requires_grad=False))
        self.constructed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for wq, wk, wv, wo in zip(self.wq, self.wk, self.wv, self.wo):
            q = x @ wq
            k = x @ wk
            v = x @ wv
            att = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
            out = out + (att @ v) @ wo
        return x + out

    def attention_pattern(self, x: torch.Tensor, head: int = 0) -> torch.Tensor:
        """Expose the softmax pattern of one head (for leakage tests)."""
        q = x @ self.wq[head]
        k = x @ self.wk[head]
        return torch.softmax(q @ k.transpose(-2, -1), dim=-1)


def build_edge_reads_source(reg: RegisterMap, cfg: Config) -> HandAttention:
    """Head: each edge token reads dist[u] from its source node.

    Query dims: [EDGE_FROM one-hot | ONES].
    Logits: edge -> node u with NODE_ID == EDGE_FROM : 4C
            anything -> sink                          : 2C
            everything else                           : 0
    Edge tokens hit their source node exactly; node/sink tokens (whose
    EDGE_FROM is zero) fall back to the sink and copy a clean 0.
    Value: DIST. Output: written to SCRATCH_SRC_DIST.
    """
    n = cfg.n_max
    c = cfg.c_match
    dk = n + 1
    wq = torch.zeros(reg.d_model, dk)
    wk = torch.zeros(reg.d_model, dk)
    wv = torch.zeros(reg.d_model, 1)
    wo = torch.zeros(1, reg.d_model)

    ef, ni = reg.slice("EDGE_FROM"), reg.slice("NODE_ID")
    for j in range(n):
        wq[ef.start + j, j] = 1.0
        wk[ni.start + j, j] = 4.0 * c
    wq[reg.idx("ONES"), n] = 1.0
    wk[reg.idx("SINK"), n] = 2.0 * c

    wv[reg.idx("DIST"), 0] = 1.0
    wo[0, reg.idx("SCRATCH_SRC_DIST")] = 1.0
    return HandAttention([(wq, wk, wv, wo)])


def build_node_gathers(reg: RegisterMap, cfg: Config) -> HandAttention:
    """D_MAX heads: node v's head k reads the candidate from its in-edge in slot k.

    Query dims: [NODE_ID one-hot | ONES].   Logits for node v's query:
        in-edge of v in slot k     : 4C   (EDGE_TO match + slot match - edge penalty)
        in-edge of v, other slot   : 2C
        node v itself (fallback)   : 3C
        edge of another node, slot k: 0;  other edges: -2C;  sink: 1C; rest: 0
    So head k routes the slot-k candidate if one exists, else dist[v] itself
    (a no-op under min). Edge/sink queries (NODE_ID = 0) fall back to the sink.
    Value: DIST + CAND (a token has at most one of them nonzero).
    Output of head k: SCRATCH_GATHER[k].
    """
    n, d = cfg.n_max, cfg.d_max
    c = cfg.c_match
    dk = n + 1
    heads = []
    et, ni = reg.slice("EDGE_TO"), reg.slice("NODE_ID")
    for k in range(d):
        wq = torch.zeros(reg.d_model, dk)
        wk = torch.zeros(reg.d_model, dk)
        wv = torch.zeros(reg.d_model, 1)
        wo = torch.zeros(1, reg.d_model)
        for j in range(n):
            wq[ni.start + j, j] = 1.0
            wk[et.start + j, j] = 4.0 * c   # destination match
            wk[ni.start + j, j] = 3.0 * c   # self fallback
        wq[reg.idx("ONES"), n] = 1.0
        wk[reg.idx_in("SLOT", k), n] = 2.0 * c   # slot match bonus
        wk[reg.idx("IS_EDGE"), n] = -2.0 * c     # edge baseline penalty
        wk[reg.idx("SINK"), n] = 1.0 * c         # sink fallback for edge queries

        wv[reg.idx("DIST"), 0] = 1.0
        wv[reg.idx("CAND"), 0] = 1.0
        wo[0, reg.idx_in("SCRATCH_GATHER", k)] = 1.0
        heads.append((wq, wk, wv, wo))
    return HandAttention(heads)
