"""Loss helpers used by autoresearch train.py."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def masked_mean(loss: torch.Tensor, mask: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if loss.shape != mask.shape:
        raise ValueError(f"loss/mask shape mismatch: {tuple(loss.shape)} vs {tuple(mask.shape)}")
    denom = mask.sum().clamp_min(float(eps))
    return (loss * mask).sum() / denom


def bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    reduction: str = "none",
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction=reduction,
    )


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = bce_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if gamma is None or float(gamma) <= 0:
        return bce
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = (1.0 - pt).pow(float(gamma))
    return focal * bce
