"""Single source of truth for the residual-stream layout.

Every weight-construction function addresses the d_model axis exclusively
through ``RegisterMap.slice(name)`` / ``RegisterMap.idx(name)``. Raw integer
indices into the residual stream are banned (enforced by a lint test).

Changing ``Config.n_max`` / ``d_max`` changes d_model here and nowhere else.
"""
from .config import Config, DEFAULT_CONFIG


class RegisterMap:
    def __init__(self, cfg: Config = DEFAULT_CONFIG):
        self.cfg = cfg
        n, d = cfg.n_max, cfg.d_max
        spec = [
            # --- structural / constant registers -------------------------
            ("ONES", 1),         # 1.0 on every real token (sink, node, edge); 0 on padding
            ("SINK", 1),         # 1.0 on the sink token only (default attention target)
            ("IS_NODE", 1),      # token-type flag
            ("IS_EDGE", 1),      # token-type flag
            # --- node-token payload --------------------------------------
            ("NODE_ID", n),      # one-hot node identity
            ("DIST", 1),         # current distance estimate (node tokens)
            # --- edge-token payload ---------------------------------------
            ("EDGE_FROM", n),    # one-hot source node of the edge
            ("EDGE_TO", n),      # one-hot destination node of the edge
            ("EDGE_W", 1),       # scalar edge weight
            ("SLOT", d),         # one-hot in-edge slot index (unique per destination)
            # --- per-round scratch (must be zero at round boundaries) ----
            ("CAND", 1),                 # candidate distance dist[u] + w (edge tokens)
            ("SCRATCH_SRC_DIST", 1),     # dist[u] copied onto the edge token
            ("SCRATCH_GATHER", d),       # candidate gathered from in-edge slot k (node tokens)
            ("SCRATCH_MIN", 1),          # running min accumulator (node tokens)
            # --- negative-cycle detection (FR7) ---------------------------
            ("SCRATCH_PREV_DIST", 1),    # DIST snapshot before the extra round
            ("FLAG_DEC", 1),             # > 0 iff DIST decreased in the extra round
            # --- spare scratch for gadget unit tests ---------------------
            ("SCRATCH_SPARE", 4),
        ]
        self._slices: dict[str, slice] = {}
        offset = 0
        for name, size in spec:
            self._slices[name] = slice(offset, offset + size)
            offset += size
        self.d_model = offset

    def names(self) -> list[str]:
        return list(self._slices)

    def slice(self, name: str) -> slice:
        return self._slices[name]

    def size(self, name: str) -> int:
        s = self._slices[name]
        return s.stop - s.start

    def idx(self, name: str) -> int:
        """Start index of a width-1 register."""
        s = self._slices[name]
        if s.stop - s.start != 1:
            raise ValueError(f"register {name!r} is not scalar")
        return s.start

    def idx_in(self, name: str, k: int) -> int:
        """Index of element k inside a multi-dim register."""
        s = self._slices[name]
        if not (0 <= k < s.stop - s.start):
            raise IndexError(f"{name}[{k}] out of range")
        return s.start + k
