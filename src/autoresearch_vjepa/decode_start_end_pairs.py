"""Global start/end pairing decoder for cycle-time estimation.

This decoder is duration-agnostic by default: it finds a globally consistent set
of non-overlapping (start, end) pairs from per-timestep start/end evidence.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class StartEndPairDecodeConfig:
    # Candidate extraction.
    candidate_min_prob: float = 0.15
    peak_min_distance_s: float = 0.4
    # Pair constraints (broad by default).
    min_pair_s: float = 0.4
    max_pair_s: Optional[float] = None
    min_gap_s: float = 0.0
    # Pair scoring.
    pair_score_bias: float = 0.0
    cycle_weight: float = 0.5
    outside_pad_s: float = 1.0
    # Interval compatibility + objective.
    allow_touching_pairs: bool = False  # if True: allow next start at previous end index.
    objective: str = "score"  # one of {"score", "count_then_score"}


@dataclass(frozen=True)
class _PairCandidate:
    start: int
    end: int
    score: float
    start_prob: float
    end_prob: float
    inside_cycle: Optional[float]
    outside_cycle: Optional[float]


def _resolve_start_end_columns(heads: Sequence[str]) -> tuple[int, int, Optional[int]]:
    h = [str(x).strip().lower() for x in heads]
    if "start" not in h or "end" not in h:
        raise ValueError(f"Paired decode requires start/end heads, got heads={list(heads)}")
    start_col = int(h.index("start"))
    end_col = int(h.index("end"))
    cycle_col = int(h.index("cycle")) if "cycle" in h else None
    return start_col, end_col, cycle_col


def _to_logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p.astype(np.float64, copy=False), 1e-4, 1.0 - 1e-4)
    return np.log(q / (1.0 - q))


def _local_peak_candidates(
    p: np.ndarray,
    *,
    min_prob: float,
    min_dist_steps: int,
) -> np.ndarray:
    n = int(p.shape[0])
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    peaks: List[int] = []
    for i in range(n):
        v = float(p[i])
        if v < float(min_prob):
            continue
        l = float(p[i - 1]) if i > 0 else -1.0
        r = float(p[i + 1]) if i < (n - 1) else -1.0
        if (v >= l and v > r) or (v > l and v >= r):
            peaks.append(int(i))
    if not peaks:
        return np.zeros((0,), dtype=np.int64)

    # NMS over 1D peaks by score descending.
    order = sorted(peaks, key=lambda i: float(p[i]), reverse=True)
    keep: List[int] = []
    min_dist = int(max(0, min_dist_steps))
    for idx in order:
        if all(abs(int(idx) - int(k)) > min_dist for k in keep):
            keep.append(int(idx))
    keep.sort()
    return np.asarray(keep, dtype=np.int64)


def _build_pair_candidates(
    *,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    p_start: np.ndarray,
    p_end: np.ndarray,
    p_cycle: Optional[np.ndarray],
    dt_ms: int,
    cfg: StartEndPairDecodeConfig,
) -> List[_PairCandidate]:
    logit_s = _to_logit(p_start)
    logit_e = _to_logit(p_end)
    min_pair_steps = int(max(1, round(float(cfg.min_pair_s) * 1000.0 / max(1, dt_ms))))
    max_pair_steps = (
        int(max(min_pair_steps, round(float(cfg.max_pair_s) * 1000.0 / max(1, dt_ms))))
        if cfg.max_pair_s is not None
        else None
    )
    outside_pad_steps = int(max(1, round(float(cfg.outside_pad_s) * 1000.0 / max(1, dt_ms))))

    out: List[_PairCandidate] = []
    if start_idx.size <= 0 or end_idx.size <= 0:
        return out

    ends_list = end_idx.tolist()
    for s in start_idx.tolist():
        # End candidates strictly after start.
        pos = bisect_right(ends_list, int(s))
        for e in ends_list[pos:]:
            dur = int(e) - int(s) + 1
            if dur < int(min_pair_steps):
                continue
            if max_pair_steps is not None and dur > int(max_pair_steps):
                break
            score = float(logit_s[int(s)] + logit_e[int(e)] - float(cfg.pair_score_bias))
            inside_cycle = None
            outside_cycle = None
            if p_cycle is not None and abs(float(cfg.cycle_weight)) > 1e-8:
                inside_cycle = float(np.mean(p_cycle[int(s) : int(e) + 1]))
                pre_lo = max(0, int(s) - int(outside_pad_steps))
                pre = p_cycle[pre_lo:int(s)]
                post_hi = min(int(p_cycle.shape[0]), int(e) + 1 + int(outside_pad_steps))
                post = p_cycle[int(e) + 1 : post_hi]
                if pre.size > 0 and post.size > 0:
                    outside_cycle = float(0.5 * (float(np.mean(pre)) + float(np.mean(post))))
                elif pre.size > 0:
                    outside_cycle = float(np.mean(pre))
                elif post.size > 0:
                    outside_cycle = float(np.mean(post))
                else:
                    outside_cycle = float(0.5)
                score += float(cfg.cycle_weight) * (float(inside_cycle) - float(outside_cycle))
            out.append(
                _PairCandidate(
                    start=int(s),
                    end=int(e),
                    score=float(score),
                    start_prob=float(p_start[int(s)]),
                    end_prob=float(p_end[int(e)]),
                    inside_cycle=(None if inside_cycle is None else float(inside_cycle)),
                    outside_cycle=(None if outside_cycle is None else float(outside_cycle)),
                )
            )
    return out


def _weighted_interval_select(
    *,
    candidates: List[_PairCandidate],
    min_gap_steps: int,
    allow_touching_pairs: bool,
    objective: str,
) -> List[_PairCandidate]:
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (int(c.end), int(c.start)))
    n = len(cands)
    min_gap = int(max(0, min_gap_steps))
    end_with_gap = [int(c.end) + int(min_gap) for c in cands]

    prev = np.full((n,), -1, dtype=np.int64)
    for i, c in enumerate(cands):
        key = int(c.start) if bool(allow_touching_pairs) else int(c.start) - 1
        j = int(bisect_right(end_with_gap, key, hi=i) - 1)
        prev[i] = int(j)

    obj = str(objective or "score").strip().lower()
    if obj not in {"score", "count_then_score"}:
        raise ValueError(f"Unsupported paired decode objective={objective!r}")

    take = np.zeros((n + 1,), dtype=np.int8)
    if obj == "score":
        dp = np.zeros((n + 1,), dtype=np.float64)
        for i in range(1, n + 1):
            skip_score = float(dp[i - 1])
            j = int(prev[i - 1])
            take_score = float(cands[i - 1].score) + float(dp[j + 1])
            if take_score > skip_score:
                dp[i] = take_score
                take[i] = 1
            else:
                dp[i] = skip_score
                take[i] = 0
    else:
        # Lexicographic objective: maximize pair-count first, then score.
        dp_count = np.zeros((n + 1,), dtype=np.int32)
        dp_score = np.zeros((n + 1,), dtype=np.float64)
        for i in range(1, n + 1):
            skip_count = int(dp_count[i - 1])
            skip_score = float(dp_score[i - 1])
            j = int(prev[i - 1])
            take_count = int(1 + int(dp_count[j + 1]))
            take_score = float(cands[i - 1].score) + float(dp_score[j + 1])
            if (take_count > skip_count) or (take_count == skip_count and take_score > skip_score):
                dp_count[i] = int(take_count)
                dp_score[i] = float(take_score)
                take[i] = 1
            else:
                dp_count[i] = int(skip_count)
                dp_score[i] = float(skip_score)
                take[i] = 0

    sel: List[_PairCandidate] = []
    i = n
    while i > 0:
        if int(take[i]) == 1:
            sel.append(cands[i - 1])
            i = int(prev[i - 1]) + 1
        else:
            i -= 1
    sel.reverse()
    return sel


def decode_start_end_pairs(
    *,
    probs: np.ndarray,  # [T, C]
    timestamps_ms: np.ndarray,  # [T]
    heads: Sequence[str],
    cfg: Optional[StartEndPairDecodeConfig] = None,
) -> Dict[str, object]:
    if probs.ndim != 2:
        raise ValueError(f"Expected probs shape [T,C], got {probs.shape}")
    if timestamps_ms.ndim != 1:
        raise ValueError(f"Expected timestamps shape [T], got {timestamps_ms.shape}")
    if int(probs.shape[0]) != int(timestamps_ms.shape[0]):
        raise ValueError("probs/timestamps length mismatch")

    T = int(probs.shape[0])
    if T <= 0:
        return {
            "pairs": [],
            "start_indices": np.zeros((0,), dtype=np.int64),
            "end_indices": np.zeros((0,), dtype=np.int64),
            "start_timestamps_ms": np.zeros((0,), dtype=np.int64),
            "end_timestamps_ms": np.zeros((0,), dtype=np.int64),
            "event_indices": np.zeros((0,), dtype=np.int64),
            "event_timestamps_ms": np.zeros((0,), dtype=np.int64),
            "cycle_count": np.zeros((0,), dtype=np.int32),
            "active_mask": np.zeros((0,), dtype=np.int8),
            "score_head": "end",
            "num_candidates": 0,
        }

    cfg = cfg or StartEndPairDecodeConfig()
    start_col, end_col, cycle_col = _resolve_start_end_columns(heads)
    p_start = probs[:, int(start_col)].astype(np.float32, copy=False)
    p_end = probs[:, int(end_col)].astype(np.float32, copy=False)
    p_cycle = probs[:, int(cycle_col)].astype(np.float32, copy=False) if cycle_col is not None else None

    dt_ms = int(np.median(np.diff(timestamps_ms))) if T > 1 else 400
    peak_min_dist_steps = int(max(0, round(float(cfg.peak_min_distance_s) * 1000.0 / max(1, dt_ms))))
    min_gap_steps = int(max(0, round(float(cfg.min_gap_s) * 1000.0 / max(1, dt_ms))))

    start_idx = _local_peak_candidates(
        p_start,
        min_prob=float(cfg.candidate_min_prob),
        min_dist_steps=int(peak_min_dist_steps),
    )
    end_idx = _local_peak_candidates(
        p_end,
        min_prob=float(cfg.candidate_min_prob),
        min_dist_steps=int(peak_min_dist_steps),
    )

    cands = _build_pair_candidates(
        start_idx=start_idx,
        end_idx=end_idx,
        p_start=p_start,
        p_end=p_end,
        p_cycle=p_cycle,
        dt_ms=int(dt_ms),
        cfg=cfg,
    )
    selected = _weighted_interval_select(
        candidates=cands,
        min_gap_steps=int(min_gap_steps),
        allow_touching_pairs=bool(cfg.allow_touching_pairs),
        objective=str(cfg.objective),
    )

    starts = np.asarray([int(c.start) for c in selected], dtype=np.int64)
    ends = np.asarray([int(c.end) for c in selected], dtype=np.int64)
    starts_ms = np.asarray([int(timestamps_ms[i]) for i in starts.tolist()], dtype=np.int64)
    ends_ms = np.asarray([int(timestamps_ms[i]) for i in ends.tolist()], dtype=np.int64)

    cycle_count = np.zeros((T,), dtype=np.int32)
    active_mask = np.zeros((T,), dtype=np.int8)
    if ends.size > 0:
        cnt = 0
        ptr = 0
        for i in range(T):
            while ptr < int(ends.size) and int(ends[ptr]) <= i:
                cnt += 1
                ptr += 1
            cycle_count[i] = int(cnt)
    for s, e in zip(starts.tolist(), ends.tolist()):
        if 0 <= int(s) <= int(e) < T:
            active_mask[int(s) : int(e) + 1] = 1

    pairs: List[dict] = []
    for c in selected:
        s = int(c.start)
        e = int(c.end)
        pairs.append(
            {
                "start_idx": int(s),
                "end_idx": int(e),
                "start_ms": int(timestamps_ms[s]),
                "end_ms": int(timestamps_ms[e]),
                "duration_ms": int(max(0, int(timestamps_ms[e]) - int(timestamps_ms[s]))),
                "score": float(c.score),
                "start_prob": float(c.start_prob),
                "end_prob": float(c.end_prob),
                "inside_cycle": c.inside_cycle,
                "outside_cycle": c.outside_cycle,
            }
        )

    return {
        "pairs": pairs,
        "start_indices": starts,
        "end_indices": ends,
        "start_timestamps_ms": starts_ms,
        "end_timestamps_ms": ends_ms,
        # Compatibility keys expected by existing inference plumbing.
        "event_indices": ends.astype(np.int64, copy=False),
        "event_timestamps_ms": ends_ms.astype(np.int64, copy=False),
        "cycle_count": cycle_count,
        "active_mask": active_mask,
        "score_head": "end",
        "num_candidates": int(len(cands)),
        "num_start_candidates": int(start_idx.size),
        "num_end_candidates": int(end_idx.size),
    }
