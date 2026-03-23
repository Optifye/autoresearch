"""Standalone batched preprocessing for V-JEPA clips."""

from __future__ import annotations

import os
from typing import Any, MutableMapping, Optional

import numpy as np
import torch


_SPECS = {
    "vjepa_rgb_256": {
        "crop": 256,
        "resize": int(256 * 256 / 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "vjepa_rgb_384": {
        "crop": 384,
        "resize": int(384 * 256 / 224),
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


def coerce_vjepa_preproc_id(*, record: Any, requested: str) -> str:
    return "vjepa_rgb_384" if str(getattr(record, "encoder_model", "")).strip().lower() == "giant" else "vjepa_rgb_256"


def _resolve_device_and_dtype() -> tuple[torch.device, torch.dtype]:
    requested_device = (os.getenv("AUTORESEARCH_PREPROC_DEVICE") or "cuda").strip().lower()
    if requested_device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    requested_dtype = (os.getenv("AUTORESEARCH_PREPROC_DTYPE") or "float32").strip().lower()
    if device.type == "cuda" and requested_dtype in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif device.type == "cuda" and requested_dtype in {"fp16", "float16", "half"}:
        dtype = torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _resize_dims(height: int, width: int, short_side: int) -> tuple[int, int]:
    if width < height:
        new_w = short_side
        new_h = int(round(short_side * height / max(width, 1)))
    else:
        new_h = short_side
        new_w = int(round(short_side * width / max(height, 1)))
    return int(new_h), int(new_w)


def preprocess_frames_batch(
    frames: np.ndarray,
    *,
    preproc_id: str,
    timings: Optional[MutableMapping[str, Any]] = None,
) -> torch.Tensor:
    if preproc_id not in _SPECS:
        raise KeyError(f"Unknown V-JEPA preproc_id '{preproc_id}'")
    if frames.ndim != 5 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames shape [B, T, H, W, 3], got {frames.shape}")

    spec = _SPECS[preproc_id]
    batch, clip_len, height, width, _ = frames.shape
    crop = int(spec["crop"])
    resize_short = int(spec["resize"])
    device, dtype = _resolve_device_and_dtype()

    tensor = torch.as_tensor(frames, device=device)
    tensor = tensor.permute(0, 1, 4, 2, 3).reshape(batch * clip_len, 3, height, width)
    tensor = tensor.to(dtype=dtype).mul_(1.0 / 255.0)

    target_h, target_w = _resize_dims(height, width, resize_short)
    if target_h != height or target_w != width:
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    if target_h < crop or target_w < crop:
        raise ValueError(f"Cannot center-crop {crop}x{crop} from resized {target_h}x{target_w}")
    start_h = max(0, (target_h - crop) // 2)
    start_w = max(0, (target_w - crop) // 2)
    tensor = tensor[:, :, start_h : start_h + crop, start_w : start_w + crop]

    mean = torch.tensor(spec["mean"], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(spec["std"], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    out = tensor.reshape(batch, clip_len, 3, crop, crop).permute(0, 2, 1, 3, 4).contiguous()
    if timings is not None:
        timings["backend"] = f"torch_{device.type}"
    return out
