"""Trainable twins of the hand-constructed Bellman-Ford transformer.

The modules here deliberately preserve the constructed model's token stream,
register layout, sublayer order, parameter shapes, residual updates, and lack
of normalisation.  The only architectural change is that parameters require
gradients.  A tied model reuses one round block; an untied model owns one block
per round.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from bf_transformer.blocks.attention import HandAttention
from bf_transformer.blocks.minred import HandMLP
from bf_transformer.blocks.round import BFRound, build_round
from bf_transformer.config import Config
from bf_transformer.model import BFTransformer
from bf_transformer.registers import RegisterMap

from .config import TRAIN_CONFIG


class TrainableAttention(nn.Module):
    """The exact ``HandAttention`` computation with trainable parameters."""

    def __init__(
        self,
        heads: Iterable[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ],
    ) -> None:
        super().__init__()
        head_list = list(heads)
        self.wq = nn.ParameterList(
            nn.Parameter(wq.detach().clone()) for wq, _, _, _ in head_list
        )
        self.wk = nn.ParameterList(
            nn.Parameter(wk.detach().clone()) for _, wk, _, _ in head_list
        )
        self.wv = nn.ParameterList(
            nn.Parameter(wv.detach().clone()) for _, _, wv, _ in head_list
        )
        self.wo = nn.ParameterList(
            nn.Parameter(wo.detach().clone()) for _, _, _, wo in head_list
        )

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(stream)
        for wq, wk, wv, wo in zip(self.wq, self.wk, self.wv, self.wo):
            query = stream @ wq
            key = stream @ wk
            value = stream @ wv
            pattern = torch.softmax(query @ key.transpose(-2, -1), dim=-1)
            delta = delta + (pattern @ value) @ wo
        return stream + delta

    def attention_pattern(
        self,
        stream: torch.Tensor,
        head: int = 0,
    ) -> torch.Tensor:
        query = stream @ self.wq[head]
        key = stream @ self.wk[head]
        return torch.softmax(query @ key.transpose(-2, -1), dim=-1)


class TrainableMLP(nn.Module):
    """The exact ``HandMLP`` residual computation with trainable parameters."""

    def __init__(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
    ) -> None:
        super().__init__()
        self.w1 = nn.Parameter(w1.detach().clone())
        self.b1 = nn.Parameter(b1.detach().clone())
        self.w2 = nn.Parameter(w2.detach().clone())

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        return stream + torch.relu(stream @ self.w1 + self.b1) @ self.w2


class TrainableRound(nn.Module):
    """One architecture-identical trainable relaxation block."""

    def __init__(self, layers: list[nn.Module], names: list[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.layer_names = list(names)

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            stream = layer(stream)
        return stream

    def trace(self, stream: torch.Tensor):
        for name, layer in zip(self.layer_names, self.layers):
            stream = layer(stream)
            yield name, layer, stream


def _clone_trainable_layer(layer: nn.Module) -> nn.Module:
    if isinstance(layer, HandAttention):
        return TrainableAttention(
            zip(layer.wq, layer.wk, layer.wv, layer.wo, strict=True)
        )
    if isinstance(layer, HandMLP):
        return TrainableMLP(layer.w1, layer.b1, layer.w2)
    raise TypeError(f"unsupported constructed layer: {type(layer).__name__}")


def _build_trainable_round(reg: RegisterMap, cfg: Config) -> TrainableRound:
    template: BFRound = build_round(reg, cfg)
    layers = [_clone_trainable_layer(layer) for layer in template.layers]
    return TrainableRound(layers, template.layer_names)


def _randomise_parameter(parameter: nn.Parameter) -> None:
    if parameter.ndim < 2:
        nn.init.zeros_(parameter)
    else:
        # Fifteen residual applications amplify ordinary Xavier initialisation.
        # A fixed small gain keeps the no-normalisation primary architecture
        # finite without changing its hypothesis space.
        nn.init.xavier_uniform_(parameter, gain=0.1)


class TrainableBFTransformer(nn.Module):
    """Tied or untied trainable model in the constructed hypothesis space."""

    def __init__(
        self,
        cfg: Config = TRAIN_CONFIG,
        *,
        tied: bool,
        random_initialisation: bool = True,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.reg = RegisterMap(cfg)
        self.tied = tied
        if tied:
            self.block = _build_trainable_round(self.reg, cfg)
        else:
            self.blocks = nn.ModuleList(
                _build_trainable_round(self.reg, cfg)
                for _ in range(cfg.rounds)
            )
        if random_initialisation:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self.parameters():
                _randomise_parameter(parameter)

    def _round_blocks(self, rounds: int):
        if rounds < 0:
            raise ValueError("rounds must be non-negative")
        if self.tied:
            return (self.block for _ in range(rounds))
        if rounds > len(self.blocks):
            raise ValueError(
                f"untied model has {len(self.blocks)} blocks, cannot run {rounds}"
            )
        return iter(self.blocks[:rounds])

    def apply_round(
        self,
        stream: torch.Tensor,
        round_index: int = 0,
    ) -> torch.Tensor:
        if self.tied:
            return self.block(stream)
        return self.blocks[round_index](stream)

    def forward(
        self,
        stream: torch.Tensor,
        rounds: int | None = None,
    ) -> torch.Tensor:
        round_count = self.cfg.rounds if rounds is None else rounds
        for block in self._round_blocks(round_count):
            stream = block(stream)
        return stream

    def forward_with_states(
        self,
        stream: torch.Tensor,
        rounds: int | None = None,
    ) -> list[torch.Tensor]:
        """Return the initial stream and every subsequent round boundary."""
        round_count = self.cfg.rounds if rounds is None else rounds
        states = [stream]
        for block in self._round_blocks(round_count):
            stream = block(stream)
            states.append(stream)
        return states

    def load_constructed_weights(
        self,
        constructed: BFTransformer,
    ) -> None:
        """Load the exact circuit, proving it lies in this hypothesis space."""
        if constructed.cfg != self.cfg:
            raise ValueError(
                "constructed and trainable models must use the same Config"
            )
        if self.tied:
            self.block.load_state_dict(constructed.block.state_dict(), strict=True)
            return
        for block in self.blocks:
            block.load_state_dict(constructed.block.state_dict(), strict=True)
