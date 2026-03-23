"""Cycle boundary label utilities for boundary-TCN training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class CycleInterval:
    # Boundary times on the video timeline (milliseconds). Treated as inclusive boundaries.
    start_ms: int
    end_ms: int


@dataclass
class CycleLabels:
    video_id: str
    camera_id: str
    fps: Optional[float]
    # Controls which timesteps are supervised.
    # - "span" (default): supervise the entire slice [first_start, last_end] (plus optional supervised_start/end overrides).
    # - "cycles_only": supervise only timesteps that fall inside labeled cycles (everything else is ignored).
    supervised_mode: str
    # Optional override for the supervised span start/end on the video timeline.
    # If set, training will include timesteps starting at this time even if no cycles are labeled yet.
    supervised_start_ms: Optional[int]
    supervised_end_ms: Optional[int]
    cycles: List[CycleInterval]


@dataclass
class MappedCycle:
    start_ms: int
    end_ms: int
    start_idx: int
    end_idx: int
    start_err_ms: int
    end_err_ms: int


@dataclass
class MappingStats:
    count: int
    mean_abs_start_err_ms: float
    mean_abs_end_err_ms: float
    max_abs_start_err_ms: int
    max_abs_end_err_ms: int


def load_cycle_labels(path: Path) -> CycleLabels:
    data = json.loads(path.read_text(encoding="utf-8"))
    cycles: List[CycleInterval] = []
    fps = data.get("fps")
    supervised_mode_raw = data.get("supervised_mode")
    if supervised_mode_raw is None:
        supervised_mode_raw = data.get("supervision")
    if supervised_mode_raw is None and bool(data.get("cycles_only")):
        supervised_mode_raw = "cycles_only"
    supervised_mode = str(supervised_mode_raw or "span").strip().lower()
    supervised_mode = supervised_mode.replace("-", "_")
    if supervised_mode in {"default", "full", "full_span"}:
        supervised_mode = "span"
    if supervised_mode not in {"span", "cycles_only"}:
        raise ValueError(f"Unsupported supervised_mode={supervised_mode!r} in {path}")

    supervised_start_ms = data.get("supervised_start_ms")
    supervised_end_ms = data.get("supervised_end_ms")
    supervised_start_frame = data.get("supervised_start_frame")
    supervised_end_frame = data.get("supervised_end_frame")

    if supervised_start_ms is None and supervised_start_frame is not None:
        if fps is None:
            raise ValueError("fps required when supervised_start_frame is provided")
        supervised_start_ms = int(round(float(supervised_start_frame) * 1000.0 / float(fps)))
    if supervised_end_ms is None and supervised_end_frame is not None:
        if fps is None:
            raise ValueError("fps required when supervised_end_frame is provided")
        supervised_end_ms = int(round(float(supervised_end_frame) * 1000.0 / float(fps)))

    for item in data.get("cycles", []):
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        if start_ms is not None and end_ms is not None:
            start_ms_i = int(start_ms)
            end_ms_i = int(end_ms)
        else:
            # Optional: allow frame-based boundaries when fps is provided.
            start_frame = item.get("start_frame")
            end_frame = item.get("end_frame")
            if start_frame is None or end_frame is None:
                continue
            if fps is None:
                raise ValueError("fps required when start_frame/end_frame are provided")
            start_ms_i = int(round(float(start_frame) * 1000.0 / float(fps)))
            # Boundaries are inclusive; convert end_frame to the ms timestamp of that frame.
            end_ms_i = int(round(float(end_frame) * 1000.0 / float(fps)))
        if end_ms_i <= start_ms_i:
            continue
        cycles.append(CycleInterval(start_ms=start_ms_i, end_ms=end_ms_i))
    return CycleLabels(
        video_id=str(data.get("video_id") or "unknown"),
        camera_id=str(data.get("camera_id") or "unknown"),
        fps=float(fps) if fps is not None else None,
        supervised_mode=str(supervised_mode),
        supervised_start_ms=int(supervised_start_ms) if supervised_start_ms is not None else None,
        supervised_end_ms=int(supervised_end_ms) if supervised_end_ms is not None else None,
        cycles=cycles,
    )


def _normalize_boundary_index_mode(mode: Optional[str]) -> str:
    token = str(mode or "legacy").strip().lower().replace("-", "_")
    if token in {"legacy", "nearest", "ordered_nearest"}:
        return token
    raise ValueError(
        f"Unsupported boundary_index_mode={mode!r}; expected one of ['legacy', 'nearest', 'ordered_nearest']"
    )


def _nearest_index(timestamps_ms: np.ndarray, target_ms: int) -> int:
    pos = int(np.searchsorted(timestamps_ms, int(target_ms), side="left"))
    if pos <= 0:
        return 0
    if pos >= int(timestamps_ms.shape[0]):
        return int(timestamps_ms.shape[0]) - 1
    left = int(pos - 1)
    right = int(pos)
    dl = abs(int(timestamps_ms[left]) - int(target_ms))
    dr = abs(int(timestamps_ms[right]) - int(target_ms))
    return right if dr < dl else left


def _start_abs_err_ms(timestamps_ms: np.ndarray, *, idx: int, start_ms: int) -> int:
    return abs(int(timestamps_ms[int(idx)]) - int(start_ms))


def _end_abs_err_ms(timestamps_ms: np.ndarray, *, idx: int, end_ms: int) -> int:
    return abs(int(end_ms) - int(timestamps_ms[int(idx)]))


def _repair_single_cycle_strict(
    *,
    start_idx: int,
    end_idx: int,
    start_ms: int,
    end_ms: int,
    timestamps_ms: np.ndarray,
) -> Tuple[int, int]:
    """Enforce strict per-cycle ordering (start_idx < end_idx) with minimal error increase."""
    s = int(start_idx)
    e = int(end_idx)
    T = int(timestamps_ms.shape[0])
    for _ in range(6):
        if s < e:
            break
        candidates: List[Tuple[float, int, int, int]] = []
        # Move end forward.
        if (e + 1) < T:
            delta_end = float(
                _end_abs_err_ms(timestamps_ms, idx=int(e + 1), end_ms=int(end_ms))
                - _end_abs_err_ms(timestamps_ms, idx=int(e), end_ms=int(end_ms))
            )
            candidates.append((delta_end, 0, int(s), int(e + 1)))
        # Move start backward.
        if (s - 1) >= 0:
            delta_start = float(
                _start_abs_err_ms(timestamps_ms, idx=int(s - 1), start_ms=int(start_ms))
                - _start_abs_err_ms(timestamps_ms, idx=int(s), start_ms=int(start_ms))
            )
            candidates.append((delta_start, 1, int(s - 1), int(e)))
        if not candidates:
            break
        candidates.sort(key=lambda item: (float(item[0]), int(item[1])))
        _delta, _tie, s, e = candidates[0]
    return int(s), int(e)


def _repair_pair_order_min_error(
    *,
    cur: dict,
    nxt: dict,
    timestamps_ms: np.ndarray,
) -> None:
    """Enforce cur.end_idx < nxt.start_idx when source labels are strictly ordered in time."""
    T = int(timestamps_ms.shape[0])
    cur_end = int(cur["end_idx"])
    nxt_start = int(nxt["start_idx"])
    if cur_end < nxt_start:
        return

    candidates: List[Tuple[float, int, str, int]] = []

    # Candidate A: move later boundary forward => nxt.start_idx = cur_end + 1
    new_nxt_start = int(cur_end + 1)
    if new_nxt_start < T and new_nxt_start < int(nxt["end_idx"]):
        delta_start = float(
            _start_abs_err_ms(timestamps_ms, idx=int(new_nxt_start), start_ms=int(nxt["start_ms"]))
            - _start_abs_err_ms(timestamps_ms, idx=int(nxt_start), start_ms=int(nxt["start_ms"]))
        )
        candidates.append((delta_start, 0, "shift_next_start", int(new_nxt_start)))

    # Candidate B: move earlier boundary backward => cur.end_idx = nxt.start_idx - 1
    new_cur_end = int(nxt_start - 1)
    if new_cur_end >= 0 and new_cur_end > int(cur["start_idx"]):
        delta_end = float(
            _end_abs_err_ms(timestamps_ms, idx=int(new_cur_end), end_ms=int(cur["end_ms"]))
            - _end_abs_err_ms(timestamps_ms, idx=int(cur_end), end_ms=int(cur["end_ms"]))
        )
        candidates.append((delta_end, 1, "shift_cur_end", int(new_cur_end)))

    if not candidates:
        return
    candidates.sort(key=lambda item: (float(item[0]), int(item[1])))
    _delta, _tie, move, new_idx = candidates[0]
    if move == "shift_next_start":
        nxt["start_idx"] = int(new_idx)
    else:
        cur["end_idx"] = int(new_idx)


def map_cycles_to_indices(
    cycles: List[CycleInterval],
    timestamps_ms: np.ndarray,
    *,
    boundary_index_mode: str = "nearest",
) -> Tuple[List[MappedCycle], MappingStats]:
    mapped: List[MappedCycle] = []
    start_errs: List[int] = []
    end_errs: List[int] = []
    if timestamps_ms.ndim != 1:
        raise ValueError("timestamps_ms must be 1D")
    if timestamps_ms.size == 0:
        return [], MappingStats(0, 0.0, 0.0, 0, 0)
    mode = _normalize_boundary_index_mode(boundary_index_mode)

    ordered: List[dict] = []
    for cycle in cycles:
        start_ms = int(cycle.start_ms)
        end_ms = int(cycle.end_ms)
        if mode == "legacy":
            start_idx = int(np.searchsorted(timestamps_ms, start_ms, side="left"))
            # Treat end boundary as inclusive: choose last timestamp <= end_ms.
            end_idx = int(np.searchsorted(timestamps_ms, end_ms, side="right") - 1)
        else:
            start_idx = int(_nearest_index(timestamps_ms, start_ms))
            end_idx = int(_nearest_index(timestamps_ms, end_ms))
            if start_idx > end_idx:
                # Guard against degenerate inversions on very short cycles.
                start_idx = int(np.searchsorted(timestamps_ms, start_ms, side="left"))
                end_idx = int(np.searchsorted(timestamps_ms, end_ms, side="right") - 1)

        start_idx = max(0, min(start_idx, len(timestamps_ms) - 1))
        end_idx = max(0, min(end_idx, len(timestamps_ms) - 1))
        if mode == "ordered_nearest" and start_idx >= end_idx:
            start_idx, end_idx = _repair_single_cycle_strict(
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                timestamps_ms=timestamps_ms,
            )
        if start_idx > end_idx:
            continue
        if mode == "ordered_nearest" and start_idx >= end_idx:
            # Unrepairable degenerate cycle on this timestamp grid.
            continue
        ordered.append(
            {
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
            }
        )

    if mode == "ordered_nearest" and ordered:
        ordered.sort(key=lambda item: (int(item["start_ms"]), int(item["end_ms"])))
        for i in range(len(ordered) - 1):
            cur = ordered[i]
            nxt = ordered[i + 1]
            if int(cur["end_ms"]) < int(nxt["start_ms"]) and int(cur["end_idx"]) >= int(nxt["start_idx"]):
                _repair_pair_order_min_error(cur=cur, nxt=nxt, timestamps_ms=timestamps_ms)

        repaired: List[dict] = []
        for item in ordered:
            s_i = int(item["start_idx"])
            e_i = int(item["end_idx"])
            if s_i >= e_i:
                s_i, e_i = _repair_single_cycle_strict(
                    start_idx=int(s_i),
                    end_idx=int(e_i),
                    start_ms=int(item["start_ms"]),
                    end_ms=int(item["end_ms"]),
                    timestamps_ms=timestamps_ms,
                )
            if s_i >= e_i:
                continue
            item["start_idx"] = int(s_i)
            item["end_idx"] = int(e_i)
            repaired.append(item)
        ordered = repaired

        # Final safety pass: enforce strict inter-cycle ordering by dropping any
        # residual unrepairable collisions between explicitly ordered boundaries.
        filtered: List[dict] = []
        for item in ordered:
            if not filtered:
                filtered.append(item)
                continue
            prev = filtered[-1]
            if int(prev["end_ms"]) < int(item["start_ms"]) and int(prev["end_idx"]) >= int(item["start_idx"]):
                continue
            filtered.append(item)
        ordered = filtered

    for item in ordered:
        start_ms = int(item["start_ms"])
        end_ms = int(item["end_ms"])
        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        start_err = int(timestamps_ms[start_idx] - int(start_ms))
        end_err = int(int(end_ms) - timestamps_ms[end_idx])
        mapped.append(
            MappedCycle(
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                start_err_ms=int(start_err),
                end_err_ms=int(end_err),
            )
        )
        start_errs.append(abs(int(start_err)))
        end_errs.append(abs(int(end_err)))

    if mapped:
        stats = MappingStats(
            count=len(mapped),
            mean_abs_start_err_ms=float(np.mean(start_errs)),
            mean_abs_end_err_ms=float(np.mean(end_errs)),
            max_abs_start_err_ms=int(np.max(start_errs)),
            max_abs_end_err_ms=int(np.max(end_errs)),
        )
    else:
        stats = MappingStats(0, 0.0, 0.0, 0, 0)
    return mapped, stats


def _gaussian_bump(length: int, center: int, sigma: float, radius: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.float32)
    if sigma <= 0:
        out[center] = 1.0
        return out
    r = int(max(0, radius))
    start = max(0, center - r)
    end = min(length - 1, center + r)
    xs = np.arange(start, end + 1, dtype=np.float32)
    dist2 = (xs - float(center)) ** 2
    weights = np.exp(-0.5 * dist2 / float(sigma * sigma)).astype(np.float32)
    # Normalize to 1.0 at the center.
    if weights.size:
        weights = weights / float(weights.max())
    out[start : end + 1] = weights
    return out


def build_boundary_targets(
    total_len: int,
    mapped_cycles: List[MappedCycle],
    *,
    ignore_radius: int = 1,
    smooth_sigma: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build dense targets and a per-timestep mask for the start/end heads.

    Returns: (y_start, y_end, y_cycle, mask_start_end)
      - y_* are float32 arrays of shape [T] with values in [0, 1]
      - mask_start_end is float32 array of shape [T] with 1.0 where start/end loss is enabled

    Important: the ignore mask applies ONLY to the start/end heads, and it does NOT mask out
    the positive boundary timestep itself (only neighbors).
    """

    T = int(total_len)
    y_start = np.zeros(T, dtype=np.float32)
    y_end = np.zeros(T, dtype=np.float32)
    y_cycle = np.zeros(T, dtype=np.float32)
    mask_start_end = np.ones(T, dtype=np.float32)

    if T <= 0 or not mapped_cycles:
        return y_start, y_end, y_cycle, mask_start_end

    # True boundary indices (hard, unsmoothed).
    start_indices: List[int] = []
    end_indices: List[int] = []

    # Hard labels first.
    for cycle in mapped_cycles:
        s = int(cycle.start_idx)
        e = int(cycle.end_idx)
        if s < 0 or e < 0 or s >= T or e >= T or s > e:
            continue
        y_start[s] = 1.0
        y_end[e] = 1.0
        y_cycle[s : e + 1] = 1.0
        start_indices.append(s)
        end_indices.append(e)

    # Optional smoothing for sparse start/end targets.
    if smooth_sigma and float(smooth_sigma) > 0:
        sigma = float(smooth_sigma)
        radius = int(np.ceil(3.0 * sigma))
        y_start_smooth = np.zeros_like(y_start)
        y_end_smooth = np.zeros_like(y_end)
        for idx in start_indices:
            y_start_smooth = np.maximum(y_start_smooth, _gaussian_bump(T, int(idx), sigma, radius))
        for idx in end_indices:
            y_end_smooth = np.maximum(y_end_smooth, _gaussian_bump(T, int(idx), sigma, radius))
        y_start = y_start_smooth
        y_end = y_end_smooth

    # Ignore neighbors around boundaries for start/end loss only (keep boundary index itself).
    r = int(max(0, ignore_radius))
    if r > 0:
        boundary_indices = sorted(set(start_indices + end_indices))
        for b in boundary_indices:
            for dt in range(-r, r + 1):
                if dt == 0:
                    continue
                t = int(b + dt)
                if 0 <= t < T:
                    mask_start_end[t] = 0.0
        # Ensure true boundary positions are always unmasked.
        for b in boundary_indices:
            mask_start_end[int(b)] = 1.0

    return y_start, y_end, y_cycle, mask_start_end
