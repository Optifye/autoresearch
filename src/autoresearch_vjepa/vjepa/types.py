"""Small shared types for local V-JEPA helpers."""

from __future__ import annotations

from dataclasses import dataclass

GPU_REQUIRED_SENTINEL = "GPU_DECODE_REQUIRED"


@dataclass(frozen=True)
class NormalizedROI:
    x: float
    y: float
    w: float
    h: float
