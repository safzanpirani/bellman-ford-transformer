"""Diagnostic-capacity variants for the v3 learnability study.

This module is additive: the v1 trainable twin remains unchanged.  The
diagnostic model preserves the token stream, residual registers, sublayer
order, and tied/untied loop structure while allowing wider internal attention
and MLP dimensions.  Bias-free pre-LayerNorm is optional and is explicitly
non-G1-compatible.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from bf_transformer.blocks.attention import HandAttention
from bf_transformer.blocks.minred import HandMLP
from bf_transformer.blocks.round import BFRound, build_round
from bf_transformer.config import Config
from bf_transformer.model import BFTransformer
from bf_transformer.registers import RegisterMap


def _embedded_matrix(
    source: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    result = torch.zeros(shape, dtype=source.dtype)
    slices = tuple(slice(0, size) for size in source.shape)
    result[slices] = source
    return result


def _randomise_matrix(parameter: nn.Parameter, gain: float) -> None:
    nn.init.xavier_uniform_(parameter, gain=gain)


class V3Attention(nn.Module):
    """Constructed attention topology with optionally wider head dimensions."""

    def __init__(
        self,
        source: HandAttention,
        *,
        width: int | None,
        random_initialisation: bool,
        init_gain: float,
    ) -> None:
        super().__init__()
        self.base_qk_widths: list[int] = []
        self.base_value_widths: list[int] = []
        self.wq = nn.ParameterList()
        self.wk = nn.ParameterList()
        self.wv = nn.ParameterList()
        self.wo = nn.ParameterList()
        for wq, wk, wv, wo in zip(
            source.wq,
            source.wk,
            source.wv,
            source.wo,
            strict=True,
        ):
            qk_width = wq.shape[1] if width is None else max(width, wq.shape[1])
            value_width = (
                wv.shape[1] if width is None else max(width, wv.shape[1])
            )
            self.base_qk_widths.append(wq.shape[1])
            self.base_value_widths.append(wv.shape[1])
            self.wq.append(
                nn.Parameter(_embedded_matrix(wq.detach(), (wq.shape[0], qk_width)))
            )
            self.wk.append(
                nn.Parameter(_embedded_matrix(wk.detach(), (wk.shape[0], qk_width)))
            )
            self.wv.append(
                nn.Parameter(
                    _embedded_matrix(wv.detach(), (wv.shape[0], value_width))
                )
            )
            self.wo.append(
                nn.Parameter(
                    _embedded_matrix(wo.detach(), (value_width, wo.shape[1]))
                )
            )
        if random_initialisation:
            with torch.no_grad():
                for parameter in self.parameters():
                    _randomise_matrix(parameter, init_gain)

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(stream)
        for wq, wk, wv, wo in zip(
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            strict=True,
        ):
            query = stream @ wq
            key = stream @ wk
            value = stream @ wv
            pattern = torch.softmax(query @ key.transpose(-2, -1), dim=-1)
            delta = delta + (pattern @ value) @ wo
        return stream + delta

    def load_constructed(self, source: HandAttention) -> None:
        with torch.no_grad():
            for target_list, source_list in (
                (self.wq, source.wq),
                (self.wk, source.wk),
                (self.wv, source.wv),
                (self.wo, source.wo),
            ):
                for target, value in zip(target_list, source_list, strict=True):
                    target.zero_()
                    slices = tuple(slice(0, size) for size in value.shape)
                    target[slices].copy_(value)


class V3MLP(nn.Module):
    """Constructed MLP topology with optionally wider hidden dimension."""

    def __init__(
        self,
        source: HandMLP,
        *,
        width: int | None,
        random_initialisation: bool,
        init_gain: float,
    ) -> None:
        super().__init__()
        hidden = (
            source.w1.shape[1]
            if width is None
            else max(width, source.w1.shape[1])
        )
        self.base_hidden_width = source.w1.shape[1]
        self.w1 = nn.Parameter(
            _embedded_matrix(source.w1.detach(), (source.w1.shape[0], hidden))
        )
        self.b1 = nn.Parameter(
            _embedded_matrix(source.b1.detach(), (hidden,))
        )
        self.w2 = nn.Parameter(
            _embedded_matrix(source.w2.detach(), (hidden, source.w2.shape[1]))
        )
        if random_initialisation:
            with torch.no_grad():
                _randomise_matrix(self.w1, init_gain)
                nn.init.zeros_(self.b1)
                _randomise_matrix(self.w2, init_gain)

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        return stream + torch.relu(stream @ self.w1 + self.b1) @ self.w2

    def load_constructed(self, source: HandMLP) -> None:
        with torch.no_grad():
            for target, value in (
                (self.w1, source.w1),
                (self.b1, source.b1),
                (self.w2, source.w2),
            ):
                target.zero_()
                slices = tuple(slice(0, size) for size in value.shape)
                target[slices].copy_(value)


class V3Round(nn.Module):
    """One diagnostic round with optional bias-free pre-LayerNorm."""

    def __init__(
        self,
        template: BFRound,
        *,
        d_model: int,
        width: int | None,
        layer_norm: bool,
        random_initialisation: bool,
        init_gain: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for source in template.layers:
            if isinstance(source, HandAttention):
                layer = V3Attention(
                    source,
                    width=width,
                    random_initialisation=random_initialisation,
                    init_gain=init_gain,
                )
            elif isinstance(source, HandMLP):
                layer = V3MLP(
                    source,
                    width=width,
                    random_initialisation=random_initialisation,
                    init_gain=init_gain,
                )
            else:
                raise TypeError(
                    f"unsupported constructed layer: {type(source).__name__}"
                )
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.layer_names = list(template.layer_names)
        self.layer_norm = layer_norm
        self.norms = nn.ModuleList(
            nn.LayerNorm(d_model, elementwise_affine=True, bias=False)
            if layer_norm
            else nn.Identity()
            for _ in layers
        )

    def forward(self, stream: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers, strict=True):
            if self.layer_norm:
                normalised = norm(stream)
                stream = stream + (layer(normalised) - normalised)
            else:
                stream = layer(stream)
        return stream

    def load_constructed(self, template: BFRound) -> None:
        if self.layer_norm:
            raise ValueError("LayerNorm diagnostic is not G1-compatible")
        for target, source in zip(self.layers, template.layers, strict=True):
            target.load_constructed(source)


class V3BFTransformer(nn.Module):
    """Tied/untied diagnostic model with explicit G1-compatibility metadata."""

    def __init__(
        self,
        cfg: Config,
        *,
        tied: bool,
        width: int | None,
        layer_norm: bool,
        random_initialisation: bool = True,
        init_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.reg = RegisterMap(cfg)
        self.tied = tied
        self.width = width
        self.layer_norm = layer_norm
        if tied:
            self.block = self._new_round(
                random_initialisation=random_initialisation,
                init_gain=init_gain,
            )
        else:
            self.blocks = nn.ModuleList(
                self._new_round(
                    random_initialisation=random_initialisation,
                    init_gain=init_gain,
                )
                for _ in range(cfg.rounds)
            )

    @property
    def g1_compatible(self) -> bool:
        return not self.layer_norm

    def _new_round(
        self,
        *,
        random_initialisation: bool,
        init_gain: float,
    ) -> V3Round:
        return V3Round(
            build_round(self.reg, self.cfg),
            d_model=self.reg.d_model,
            width=self.width,
            layer_norm=self.layer_norm,
            random_initialisation=random_initialisation,
            init_gain=init_gain,
        )

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
        round_count = self.cfg.rounds if rounds is None else rounds
        states = [stream]
        for block in self._round_blocks(round_count):
            stream = block(stream)
            states.append(stream)
        return states

    def load_constructed_weights(self, constructed: BFTransformer) -> None:
        if not self.g1_compatible:
            raise ValueError("LayerNorm diagnostic is not G1-compatible")
        if constructed.cfg != self.cfg:
            raise ValueError(
                "constructed and diagnostic models must use the same Config"
            )
        if self.tied:
            self.block.load_constructed(constructed.block)
        else:
            for block in self.blocks:
                block.load_constructed(constructed.block)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def cosine_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    minimum_ratio: float,
) -> float:
    """Shared diagnostic schedule with well-defined short-run behaviour."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    denominator = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine
