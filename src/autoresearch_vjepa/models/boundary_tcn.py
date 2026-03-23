"""Temporal ConvNet variants for dense boundary detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def compute_left_context(*, kernel_size: int, dilations: Iterable[int], convs_per_block: int = 2) -> int:
    k = int(kernel_size)
    if k <= 0:
        raise ValueError("kernel_size must be positive")
    convs = int(convs_per_block)
    if convs <= 0:
        raise ValueError("convs_per_block must be positive")
    dsum = int(sum(int(d) for d in dilations))
    return (k - 1) * convs * dsum


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.dilation <= 0:
            raise ValueError("dilation must be positive")
        self._pad_left = self.dilation * (self.kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            self.kernel_size,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )

    @property
    def pad_left(self) -> int:
        return self._pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_left > 0:
            x = F.pad(x, (self._pad_left, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(float(dropout))
        self.use_layernorm = bool(use_layernorm)
        self.norm1 = nn.LayerNorm(channels) if self.use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(channels) if self.use_layernorm else nn.Identity()
        self.act = nn.ReLU()

    def _apply_norm(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        if isinstance(norm, nn.Identity):
            return x
        x_t = x.transpose(1, 2)
        x_t = norm(x_t)
        return x_t.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self._apply_norm(x, self.norm1)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self._apply_norm(x, self.norm2)
        return x + residual


class SymmetricTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_layernorm: bool,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        if k <= 0 or (k % 2) == 0:
            raise ValueError("SymmetricTemporalBlock requires a positive odd kernel_size")
        pad = int(dilation) * (k - 1) // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k, dilation=int(dilation), padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, dilation=int(dilation), padding=pad)
        self.dropout = nn.Dropout(float(dropout))
        self.use_layernorm = bool(use_layernorm)
        self.norm1 = nn.LayerNorm(channels) if self.use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(channels) if self.use_layernorm else nn.Identity()
        self.act = nn.GELU()

    def _apply_norm(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        if isinstance(norm, nn.Identity):
            return x
        x_t = x.transpose(1, 2)
        x_t = norm(x_t)
        return x_t.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self._apply_norm(x, self.norm1)
        x = self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self._apply_norm(x, self.norm2)
        return x + residual


@dataclass
class BoundaryTCNConfig:
    input_dim: int = 1024
    hidden_dim: int = 128
    out_dim: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    use_layernorm: bool = True
    dilations: Optional[Tuple[int, ...]] = None
    bidirectional: bool = False
    task_specific_heads: bool = False
    base_heads: int = 3

    def resolved_dilations(self) -> List[int]:
        if self.dilations is None:
            return [1, 2, 4, 8, 16, 32]
        return [int(d) for d in self.dilations]

    @property
    def left_context(self) -> int:
        if bool(self.bidirectional):
            return 0
        return compute_left_context(
            kernel_size=self.kernel_size,
            dilations=self.resolved_dilations(),
            convs_per_block=2,
        )


class BoundaryTCN(nn.Module):
    def __init__(self, cfg: BoundaryTCNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if int(cfg.out_dim) <= 0:
            raise ValueError("out_dim must be positive")
        base_heads = int(max(1, min(int(cfg.base_heads), int(cfg.out_dim))))
        extra_heads = int(max(0, int(cfg.out_dim) - int(base_heads)))
        self.base_heads = int(base_heads)
        self.extra_heads = int(extra_heads)
        self.in_proj = nn.Conv1d(cfg.input_dim, cfg.hidden_dim, kernel_size=1)
        block_cls = SymmetricTemporalBlock if bool(cfg.bidirectional) else TemporalBlock
        blocks = [
            block_cls(
                cfg.hidden_dim,
                kernel_size=cfg.kernel_size,
                dilation=dilation,
                dropout=cfg.dropout,
                use_layernorm=cfg.use_layernorm,
            )
            for dilation in cfg.resolved_dilations()
        ]
        self.blocks = nn.Sequential(*blocks)
        if bool(cfg.task_specific_heads):
            self.base_refines = nn.ModuleList(
                [
                    block_cls(
                        cfg.hidden_dim,
                        kernel_size=cfg.kernel_size,
                        dilation=1,
                        dropout=cfg.dropout,
                        use_layernorm=cfg.use_layernorm,
                    )
                    for _ in range(self.base_heads)
                ]
            )
            self.base_out = nn.ModuleList([nn.Conv1d(cfg.hidden_dim, 1, kernel_size=1) for _ in range(self.base_heads)])
            self.out_proj = nn.Conv1d(cfg.hidden_dim, int(self.extra_heads), kernel_size=1) if self.extra_heads > 0 else None
        else:
            self.base_refines = None
            self.base_out = None
            self.out_proj = nn.Conv1d(cfg.hidden_dim, int(cfg.out_dim), kernel_size=1)

    @property
    def shared_blocks(self) -> nn.Sequential:
        return self.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(x.shape)}")
        x = x.transpose(1, 2)
        x = self.in_proj(x)
        x = self.blocks(x)
        if self.base_refines is not None and self.base_out is not None:
            outputs = [head(refine(x)) for refine, head in zip(self.base_refines, self.base_out)]
            if self.out_proj is not None:
                outputs.append(self.out_proj(x))
            x = torch.cat(outputs, dim=1)
        else:
            x = self.out_proj(x)
        return x.transpose(1, 2)
