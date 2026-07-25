"""Hand-constructed ReLU machinery: the min gadget and exact copy/clear.

Identities used (all float-exact):
    identity:  x = ReLU(x) - ReLU(-x)
    min:       min(a, b) = a - ReLU(a - b)

A HandMLP computes  x + ReLU(x @ W1 + b1) @ W2  (pure residual delta-write).
"""
import torch
from torch import nn


class HandMLP(nn.Module):
    def __init__(self, w1: torch.Tensor, b1: torch.Tensor, w2: torch.Tensor):
        super().__init__()
        self.w1 = nn.Parameter(w1, requires_grad=False)
        self.b1 = nn.Parameter(b1, requires_grad=False)
        self.w2 = nn.Parameter(w2, requires_grad=False)
        self.constructed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(x @ self.w1 + self.b1) @ self.w2


class MLPBuilder:
    """Assemble a HandMLP one hidden unit at a time.

    Each unit is: h = ReLU(sum_i w_i * x[in_i] + bias), contributing
    sum_j v_j * h to output dims out_j (added to the residual stream).
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self._units: list[tuple[dict[int, float], float, dict[int, float]]] = []

    def unit(self, ins: dict[int, float], bias: float, outs: dict[int, float]) -> "MLPBuilder":
        self._units.append((dict(ins), float(bias), dict(outs)))
        return self

    def copy(self, src: int, dst: int, sign: float = 1.0) -> "MLPBuilder":
        """dst += sign * x[src], exact for any sign of x[src] (ReLU pair)."""
        self.unit({src: 1.0}, 0.0, {dst: sign})
        self.unit({src: -1.0}, 0.0, {dst: -sign})
        return self

    def clear(self, src: int) -> "MLPBuilder":
        """src += -x[src]  (i.e. zero the register, whatever its sign)."""
        return self.copy(src, src, sign=-1.0)

    def min_fold(self, acc: int, src: int) -> "MLPBuilder":
        """acc <- min(acc, src):   acc += -ReLU(acc - src)."""
        self.unit({acc: 1.0, src: -1.0}, 0.0, {acc: -1.0})
        return self

    def build(self) -> HandMLP:
        h = len(self._units)
        w1 = torch.zeros(self.d_model, h, dtype=torch.float32)
        b1 = torch.zeros(h, dtype=torch.float32)
        w2 = torch.zeros(h, self.d_model, dtype=torch.float32)
        for col, (ins, bias, outs) in enumerate(self._units):
            for i, w in ins.items():
                w1[i, col] = w
            b1[col] = bias
            for j, v in outs.items():
                w2[col, j] = v
        return HandMLP(w1, b1, w2)


def build_min_fold_layer(d_model: int, acc: int, src: int) -> HandMLP:
    """Standalone pairwise-min layer (used directly by the gadget tests)."""
    return MLPBuilder(d_model).min_fold(acc, src).build()


def build_copy_layer(d_model: int, src: int, dst: int, sign: float = 1.0) -> HandMLP:
    return MLPBuilder(d_model).copy(src, dst, sign).build()
