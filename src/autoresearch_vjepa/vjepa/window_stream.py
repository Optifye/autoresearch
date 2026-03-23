"""Streaming window generator for edge mining.

Windowing/IDs are intentionally kept identical to `cexp.clip_extractor.stream_clips`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .id_utils import clip_id_for, video_id_from_path
from .roi import apply_roi
from .types import NormalizedROI
from .video_reader import VideoStream


@dataclass(frozen=True)
class VideoInfo:
    video_id: str
    fps: float
    fps_effective: float
    step: int
    width: Optional[int]
    height: Optional[int]


@dataclass(frozen=True)
class WindowRef:
    window_idx: int
    clip_id: str
    start_frame: int
    end_frame: int
    t_start_sec: float
    t_end_sec: float
    frames: List[np.ndarray]


def iter_windows(
    *,
    video_path: Path,
    camera_hint: Optional[str] = None,
    roi: Optional[dict],
    clip_len: int,
    stride: int,
    frame_skip: int,
    decode_device: str,
    decode_gpu_id: int,
    prefer_decord: bool = True,
    strict_gpu: bool = False,
    start_offset_samples: int = 0,
) -> tuple[VideoInfo, Iterator[WindowRef]]:
    """Yield ROI-cropped windows as lists of RGB frames.

    start_frame/end_frame refer to the *source* frame timeline (i.e. include frame_skip).
    """

    video_path = Path(video_path)
    video_id = video_id_from_path(video_path, camera_hint=camera_hint)
    step = max(1, int(frame_skip) + 1)
    if clip_len <= 0 or stride <= 0:
        raise ValueError(f"clip_len and stride must be > 0 (got clip_len={clip_len}, stride={stride})")

    stream = VideoStream(
        video_path,
        prefer_decord=prefer_decord,
        target_fps=None,
        frame_skip=frame_skip,
        decode_device=decode_device,
        decode_gpu_id=decode_gpu_id,
        strict_gpu=bool(strict_gpu),
        log_decodes=True,
    )
    meta = stream.metadata
    fps = float(meta.fps or 30.0)
    fps_effective = fps / float(step)

    roi_cfg: Optional[NormalizedROI] = None
    if roi:
        try:
            roi_cfg = NormalizedROI(x=float(roi["x"]), y=float(roi["y"]), w=float(roi["w"]), h=float(roi["h"]))
        except Exception:
            roi_cfg = None

    info = VideoInfo(
        video_id=video_id,
        fps=fps,
        fps_effective=fps_effective,
        step=step,
        width=meta.width,
        height=meta.height,
    )

    def _iter() -> Iterator[WindowRef]:
        # Important: `stride` is the step size between *window starts* in the sampled
        # (post-frame_skip) timeline. The previous buffer-slicing implementation
        # (`buffer = buffer[stride:]`) accidentally produced a step size of `clip_len`
        # whenever `stride > clip_len` (because slicing past the end clears the buffer
        # without skipping additional stream frames). That breaks expected cadences like
        # stride=10 @ 25fps -> ~2.5Hz when clip_len=4.
        #
        # We instead schedule windows explicitly at end_sample indices:
        #   end_sample = (clip_len - 1) + k * stride
        buffer: List[np.ndarray] = []
        window_idx = 0
        offset = max(0, int(start_offset_samples))
        next_end_sample = (int(clip_len) - 1) + offset
        for idx, frame in enumerate(stream):
            buffer.append(frame)
            if idx < next_end_sample:
                continue
            if idx != next_end_sample:
                # Maintain a trailing buffer of `clip_len` frames.
                if len(buffer) > clip_len:
                    buffer = buffer[-clip_len:]
                continue
            if len(buffer) > clip_len:
                buffer = buffer[-clip_len:]
            start_sample = idx - clip_len + 1
            end_sample = idx
            start_frame = start_sample * step
            end_frame = end_sample * step
            clip_id = clip_id_for(video_id, start_frame, end_frame)

            processed = apply_roi(buffer[:clip_len], roi_cfg, resize_to=None)
            even_h = processed[0].shape[0] - (processed[0].shape[0] % 2)
            even_w = processed[0].shape[1] - (processed[0].shape[1] % 2)
            if even_h <= 0 or even_w <= 0:
                next_end_sample += int(stride)
                continue
            if even_h != processed[0].shape[0] or even_w != processed[0].shape[1]:
                processed = [f[:even_h, :even_w] for f in processed]

            t_start_sec = float(start_frame) / fps
            # Treat the interval as end-exclusive for better time-bin behavior.
            t_end_sec = float(end_frame + step) / fps
            yield WindowRef(
                window_idx=window_idx,
                clip_id=clip_id,
                start_frame=start_frame,
                end_frame=end_frame,
                t_start_sec=t_start_sec,
                t_end_sec=t_end_sec,
                frames=processed,
            )
            window_idx += 1
            next_end_sample += int(stride)

    return info, _iter()
