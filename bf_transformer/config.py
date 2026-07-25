"""Global construction constants.

Everything about the construction is derived from these few numbers.
``d_model`` itself is derived from the register map (see registers.py),
never set by hand.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    n_max: int = 8        # max number of nodes
    w_max: float = 10.0   # max |edge weight|
    d_max: int = 8        # max in-degree per node (incl. self-loops / parallel edges)
    c_hard: float = 1e4   # attention logit scale; gaps of c_hard/4 make softmax exactly one-hot

    @property
    def inf_b(self) -> float:
        """Bounded infinity: provably larger than any real path length."""
        return self.n_max * self.w_max + 1.0

    @property
    def k_push(self) -> float:
        """Gain on the INF-guard ReLU: pushes a candidate whose source is at
        INF_B back above INF_B (it is then clamped to exactly INF_B)."""
        return 4.0 * (self.w_max + 1.0)

    @property
    def c_match(self) -> float:
        """Base logit unit. All attention logit gaps are >= c_match."""
        return self.c_hard / 4.0

    @property
    def rounds(self) -> int:
        """Bellman-Ford needs n-1 synchronous relaxation rounds."""
        return self.n_max - 1

    @property
    def e_max(self) -> int:
        """Max edges: every node at full in-degree."""
        return self.n_max * self.d_max

    @property
    def seq_len(self) -> int:
        """1 sink token + node tokens + edge tokens."""
        return 1 + self.n_max + self.e_max

    def __post_init__(self) -> None:
        if self.n_max < 2:
            raise ValueError("n_max must be >= 2")
        if self.d_max < 1:
            raise ValueError("d_max must be >= 1")
        if self.w_max <= 0:
            raise ValueError("w_max must be > 0")
        # All ReLU pre-activations stay comfortably inside float32 exact-integer
        # range (and far from overflow): values are bounded by inf_b + w_max.
        if not (self.inf_b + self.w_max < 1e6):
            raise ValueError("inf_b + w_max must stay < 1e6 (float32 comfort zone)")
        # softmax is *exactly* one-hot only when exp(-gap) underflows float32.
        # exp(x) == 0.0 in float32 for x < -104; we require a wide margin.
        if not (self.c_match > 200.0):
            raise ValueError("c_hard/4 must exceed 200 so off-target softmax mass underflows to exactly 0")


DEFAULT_CONFIG = Config()
