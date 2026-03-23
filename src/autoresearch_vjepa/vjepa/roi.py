"""ROI helpers for normalized bounding boxes."""

from __future__ import annotations

from typing import List, Optional, Tuple

try:  # pragma: no cover
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None
import numpy as np

from .types import NormalizedROI


def roi_to_pixels(roi: NormalizedROI, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(roi.x, 1.0)) * width)
    y1 = int(max(0, min(roi.y, 1.0)) * height)
    w = int(max(0, min(roi.w, 1.0)) * width)
    h = int(max(0, min(roi.h, 1.0)) * height)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height resolve to zero pixels")
    x2 = min(width, x1 + w)
    y2 = min(height, y1 + h)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def apply_roi(
    frames: List[np.ndarray],
    roi: Optional[NormalizedROI],
    *,
    resize_to: Optional[int] = None,
) -> List[np.ndarray]:
    if roi is None:
        cropped = frames
    else:
        height, width = frames[0].shape[0], frames[0].shape[1]
        x1, y1, w, h = roi_to_pixels(roi, width, height)
        cropped = [frame[y1 : y1 + h, x1 : x1 + w] for frame in frames]
    if resize_to is None:
        return cropped
    if cv2 is None:
        raise RuntimeError("cv2 is required for apply_roi(resize_to=...) but is not installed")
    resized: List[np.ndarray] = []
    for frame in cropped:
        resized.append(cv2.resize(frame, (resize_to, resize_to), interpolation=cv2.INTER_CUBIC))
    return resized
