"""Convert dense run-config labels into boundary-training label shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .contracts import DenseVideo


LABEL_IDLE = 0
LABEL_ACTION = 1
LABEL_IGNORE = 2


@dataclass(frozen=True)
class DenseLabelShard:
    shard_id: str
    video_id: str
    camera_id: str
    supervised_start_frame: int
    supervised_end_frame: int
    cycles: Tuple[Tuple[int, int], ...]
    class_targets_rle: Tuple[Tuple[int, int, int], ...] = tuple()
    class_label_map: Dict[str, int] = field(default_factory=dict)
    action_labels: Tuple[Tuple[str, int, Optional[str]], ...] = tuple()
    ignore_label: int = LABEL_IGNORE
    fallback_idle_label: int = LABEL_IDLE

    def to_cycle_labels_json(self, *, fps: float) -> Dict[str, object]:
        cycles = [
            {"start_frame": int(s), "end_frame": int(e)}
            for s, e in self.cycles
        ]
        action_labels = [
            {
                "label_name": str(label_name),
                "label_id": int(label_id),
                "action_class_id": str(action_class_id) if action_class_id is not None else None,
            }
            for label_name, label_id, action_class_id in self.action_labels
        ]
        return {
            "video_id": str(self.video_id),
            "camera_id": str(self.camera_id),
            "fps": float(fps),
            "supervised_mode": "span",
            "supervised_start_frame": int(self.supervised_start_frame),
            "supervised_end_frame": int(self.supervised_end_frame),
            "cycles": cycles,
            # Dense-v4 metadata consumed by multiclass TCN training.
            "label_map": {str(k): int(v) for k, v in dict(self.class_label_map).items()},
            "fallback_idle_label": int(self.fallback_idle_label),
            "ignore_label": int(self.ignore_label),
            "action_labels": action_labels,
            "class_targets_rle": [
                [int(label_id), int(start_frame), int(end_frame)]
                for label_id, start_frame, end_frame in self.class_targets_rle
            ],
        }


def _merge_spans(spans: Iterable[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    ordered = sorted((int(lbl), int(s), int(e)) for lbl, s, e in spans if int(e) >= int(s))
    if not ordered:
        return []
    out: List[Tuple[int, int, int]] = []
    cur_lbl, cur_s, cur_e = ordered[0]
    for lbl, s, e in ordered[1:]:
        if lbl == cur_lbl and s <= (cur_e + 1):
            cur_e = max(cur_e, e)
            continue
        out.append((cur_lbl, cur_s, cur_e))
        cur_lbl, cur_s, cur_e = lbl, s, e
    out.append((cur_lbl, cur_s, cur_e))
    return out


def _max_end(video: DenseVideo) -> int:
    max_end = -1
    for s, e in video.cycles:
        max_end = max(max_end, int(e))
    for s, e in video.ignore_regions:
        max_end = max(max_end, int(e))
    for _label_type, _label_name, _action_class_id, s, e in video.regions:
        max_end = max(max_end, int(e))
    for _label, s, e in video.frame_labels_rle:
        max_end = max(max_end, int(e))
    if video.num_frames is not None and int(video.num_frames) > 0:
        max_end = max(max_end, int(video.num_frames) - 1)
    if max_end < 0:
        max_end = 0
    return int(max_end)


def _resolve_ignore_label(label_map: Optional[Dict[str, int]], ignore_label: Optional[int]) -> int:
    if ignore_label is not None:
        return int(ignore_label)
    if isinstance(label_map, dict) and "ignore" in label_map:
        return int(label_map["ignore"])
    return LABEL_IGNORE


def _resolve_fallback_idle_label(label_map: Optional[Dict[str, int]], fallback_idle_label: Optional[int]) -> int:
    if fallback_idle_label is not None:
        return int(fallback_idle_label)
    if isinstance(label_map, dict) and "idle" in label_map:
        return int(label_map["idle"])
    return LABEL_IDLE


def _iter_action_label_rows(action_labels: Optional[Sequence[Any]]) -> Iterable[Tuple[str, int, Optional[str]]]:
    for item in action_labels or []:
        if isinstance(item, dict):
            label_name = str(item.get("label_name") or "").strip().lower()
            label_id_raw = item.get("label_id")
            action_class_id_raw = item.get("action_class_id")
        else:
            label_name = str(getattr(item, "label_name", "") or "").strip().lower()
            label_id_raw = getattr(item, "label_id", None)
            action_class_id_raw = getattr(item, "action_class_id", None)
        if not label_name:
            continue
        try:
            label_id = int(label_id_raw)
        except (TypeError, ValueError):
            continue
        action_class_id = str(action_class_id_raw).strip() if action_class_id_raw is not None else ""
        yield str(label_name), int(label_id), (action_class_id or None)


def _resolve_action_labels(
    *,
    label_map: Optional[Dict[str, int]],
    action_labels: Optional[Sequence[Any]],
) -> List[Tuple[str, int, Optional[str]]]:
    rows: List[Tuple[str, int, Optional[str]]] = []
    seen = set()
    for label_name, label_id, action_class_id in _iter_action_label_rows(action_labels):
        key = str(label_name)
        if key in seen:
            continue
        seen.add(key)
        rows.append((str(label_name), int(label_id), action_class_id))

    if rows:
        return rows

    if isinstance(label_map, dict):
        if "action" in label_map:
            try:
                return [("action", int(label_map["action"]), None)]
            except (TypeError, ValueError):
                pass
        fallback_rows = [
            (str(name).strip().lower(), int(label_id), None)
            for name, label_id in label_map.items()
            if str(name).strip().lower() not in {"idle", "ignore"}
        ]
        fallback_rows.sort(key=lambda item: (int(item[1]), str(item[0])))
        if fallback_rows:
            return fallback_rows

    return [("action", LABEL_ACTION, None)]


def _build_labels_from_rle(
    video: DenseVideo,
    *,
    action_label_ids: Sequence[int],
    ignore_label: int,
) -> Optional[List[Tuple[int, int, int]]]:
    if not video.frame_labels_rle:
        return None
    action_ids = {int(label_id) for label_id in action_label_ids}
    ignore_label_i = int(ignore_label)
    spans: List[Tuple[int, int, int]] = []
    for label, start, end in video.frame_labels_rle:
        lbl = int(label)
        if int(end) < int(start):
            continue
        if lbl == ignore_label_i:
            canonical = LABEL_IGNORE
        elif lbl in action_ids:
            canonical = LABEL_ACTION
        else:
            canonical = LABEL_IDLE
        spans.append((int(canonical), int(start), int(end)))
    return _merge_spans(spans)


def _build_labels_from_regions(
    video: DenseVideo,
    *,
    action_label_names: Sequence[str],
) -> List[Tuple[int, int, int]]:
    max_end = _max_end(video)
    size = int(max_end + 1)
    labels = np.zeros((size,), dtype=np.uint8)
    action_label_set = {str(name).strip().lower() for name in action_label_names}

    # Base overlays from explicit ignore/cycles first.
    for s, e in video.ignore_regions:
        labels[int(s) : int(e) + 1] = LABEL_IGNORE
    for s, e in video.cycles:
        labels[int(s) : int(e) + 1] = LABEL_ACTION

    for label_type, label_name, _action_class_id, s, e in video.regions:
        token_type = str(label_type).strip().lower()
        token_name = str(label_name).strip().lower()
        if token_type == "ignore" or token_name == "ignore":
            lbl = LABEL_IGNORE
        elif token_type == "action" or token_name in action_label_set:
            lbl = LABEL_ACTION
        else:
            lbl = LABEL_IDLE
        labels[int(s) : int(e) + 1] = int(lbl)

    # Convert to RLE.
    spans: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(labels[0]) if labels.size else LABEL_IDLE
    for i in range(1, size):
        if int(labels[i]) != cur:
            spans.append((int(cur), int(start), int(i - 1)))
            start = i
            cur = int(labels[i])
    spans.append((int(cur), int(start), int(size - 1)))
    return _merge_spans(spans)


def _build_multiclass_rle(
    *,
    video: DenseVideo,
    label_map: Dict[str, int],
    action_label_rows: Sequence[Tuple[str, int, Optional[str]]],
    ignore_label: int,
    fallback_idle_label: int,
) -> List[Tuple[int, int, int]]:
    if video.frame_labels_rle:
        spans = []
        for label, start, end in video.frame_labels_rle:
            if int(end) < int(start):
                continue
            spans.append((int(label), int(start), int(end)))
        return _merge_spans(spans)

    max_end = _max_end(video)
    size = int(max_end + 1)
    labels = np.full((size,), int(fallback_idle_label), dtype=np.int64)

    fallback_action_label_id = (
        int(action_label_rows[0][1]) if action_label_rows else int(label_map.get("action", LABEL_ACTION))
    )
    action_label_by_name: Dict[str, int] = {}
    action_label_by_class_id: Dict[str, int] = {}
    for label_name, label_id, action_class_id in action_label_rows:
        token_name = str(label_name).strip().lower()
        if token_name and token_name not in action_label_by_name:
            action_label_by_name[token_name] = int(label_id)
        token_class_id = str(action_class_id or "").strip().lower()
        if token_class_id and token_class_id not in action_label_by_class_id:
            action_label_by_class_id[token_class_id] = int(label_id)

    for s, e in video.ignore_regions:
        labels[int(s) : int(e) + 1] = int(ignore_label)
    for s, e in video.cycles:
        labels[int(s) : int(e) + 1] = int(fallback_action_label_id)

    for label_type, label_name, action_class_id, s, e in video.regions:
        token_type = str(label_type).strip().lower()
        token_name = str(label_name).strip().lower()
        token_action_class_id = str(action_class_id or "").strip().lower()
        if token_type == "ignore" or token_name == "ignore":
            label_id = int(ignore_label)
        elif token_type == "action" or token_name in action_label_by_name:
            label_id = action_label_by_class_id.get(token_action_class_id)
            if label_id is None:
                label_id = action_label_by_name.get(token_name)
            if label_id is None and token_name == "action":
                raw_action_label = label_map.get("action")
                if raw_action_label is not None:
                    try:
                        label_id = int(raw_action_label)
                    except (TypeError, ValueError):
                        label_id = None
            if label_id is None:
                label_id = int(fallback_action_label_id)
        else:
            label_id = int(fallback_idle_label)
        labels[int(s) : int(e) + 1] = int(label_id)

    spans: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(labels[0]) if labels.size else int(fallback_idle_label)
    for i in range(1, size):
        if int(labels[i]) != cur:
            spans.append((int(cur), int(start), int(i - 1)))
            start = i
            cur = int(labels[i])
    spans.append((int(cur), int(start), int(size - 1)))
    return _merge_spans(spans)


def _clip_spans_to_interval(
    spans: Sequence[Tuple[int, int, int]],
    interval: Tuple[int, int],
) -> List[Tuple[int, int, int]]:
    lo = int(interval[0])
    hi = int(interval[1])
    out: List[Tuple[int, int, int]] = []
    for label_id, start, end in spans:
        if int(end) < lo or int(start) > hi:
            continue
        clipped_start = max(lo, int(start))
        clipped_end = min(hi, int(end))
        if clipped_end < clipped_start:
            continue
        out.append((int(label_id), int(clipped_start), int(clipped_end)))
    return _merge_spans(out)


def _extract_intervals(spans: Sequence[Tuple[int, int, int]], label: int) -> List[Tuple[int, int]]:
    return [(int(s), int(e)) for lbl, s, e in spans if int(lbl) == int(label)]


def _intersect_interval(a: Tuple[int, int], b: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    lo = max(int(a[0]), int(b[0]))
    hi = min(int(a[1]), int(b[1]))
    if hi < lo:
        return None
    return int(lo), int(hi)


def _normalize_temporal_structure_mode(value: Any) -> str:
    token = str(value or "cyclic").strip().lower().replace("-", "_")
    if token in {"cyclic", "event"}:
        return token
    return "cyclic"


def _extract_event_units(
    *,
    class_targets_rle: Sequence[Tuple[int, int, int]],
    ignore_label: int,
    interval: Tuple[int, int],
) -> List[Tuple[int, int]]:
    local_targets = _clip_spans_to_interval(class_targets_rle, interval)
    return [
        (int(start), int(end))
        for label_id, start, end in local_targets
        if int(label_id) != int(ignore_label) and int(end) >= int(start)
    ]


def build_dense_label_shards(
    video: DenseVideo,
    *,
    label_map: Optional[Dict[str, int]] = None,
    action_labels: Optional[Sequence[Any]] = None,
    ignore_label: Optional[int] = None,
    fallback_idle_label: Optional[int] = None,
    temporal_structure_mode: str = "cyclic",
) -> List[DenseLabelShard]:
    temporal_mode = _normalize_temporal_structure_mode(temporal_structure_mode)
    class_label_map = {str(k): int(v) for k, v in dict(label_map or {}).items()}
    if not class_label_map:
        class_label_map = {"idle": LABEL_IDLE, "action": LABEL_ACTION, "ignore": LABEL_IGNORE}

    ignore_label_i = _resolve_ignore_label(class_label_map, ignore_label)
    fallback_idle_label_i = _resolve_fallback_idle_label(class_label_map, fallback_idle_label)
    action_label_rows = _resolve_action_labels(label_map=class_label_map, action_labels=action_labels)
    action_label_ids = [int(row[1]) for row in action_label_rows]
    action_label_names = [str(row[0]) for row in action_label_rows]

    spans = _build_labels_from_rle(
        video,
        action_label_ids=action_label_ids,
        ignore_label=int(ignore_label_i),
    )
    if spans is None:
        spans = _build_labels_from_regions(
            video,
            action_label_names=action_label_names,
        )

    class_targets_rle = _build_multiclass_rle(
        video=video,
        label_map=class_label_map,
        action_label_rows=action_label_rows,
        ignore_label=int(ignore_label_i),
        fallback_idle_label=int(fallback_idle_label_i),
    )

    supervised = _extract_intervals(spans, LABEL_IDLE) + _extract_intervals(spans, LABEL_ACTION)
    supervised = sorted(supervised)
    if not supervised:
        return []

    explicit_cycles: List[Tuple[int, int]] = []
    for cyc in sorted(video.cycles):
        s_i = int(cyc[0])
        e_i = int(cyc[1])
        if e_i < s_i:
            continue
        pair = (s_i, e_i)
        if explicit_cycles and explicit_cycles[-1] == pair:
            continue
        explicit_cycles.append(pair)

    # Preserve explicit cycle separators when provided by annotation payloads.
    # Fall back to canonical action spans only for legacy payloads without video.cycles.
    cycles = explicit_cycles if explicit_cycles else sorted(_extract_intervals(spans, LABEL_ACTION))
    if temporal_mode == "cyclic" and not cycles:
        return []

    # supervised contains idle+action intervals which are already contiguous by label;
    # merge across idle/action boundaries where there is no ignore gap.
    merged_supervised: List[Tuple[int, int]] = []
    for s, e in supervised:
        if not merged_supervised:
            merged_supervised.append((int(s), int(e)))
            continue
        prev_s, prev_e = merged_supervised[-1]
        if int(s) <= int(prev_e) + 1:
            merged_supervised[-1] = (int(prev_s), max(int(prev_e), int(e)))
        else:
            merged_supervised.append((int(s), int(e)))

    shards: List[DenseLabelShard] = []
    for idx, interval in enumerate(merged_supervised):
        local_class_targets = _clip_spans_to_interval(class_targets_rle, interval)
        if temporal_mode == "event":
            local_cycles = _extract_event_units(
                class_targets_rle=class_targets_rle,
                ignore_label=int(ignore_label_i),
                interval=interval,
            )
        else:
            local_cycles = []
            for cyc in cycles:
                inter = _intersect_interval(cyc, interval)
                if inter is None:
                    continue
                local_cycles.append((int(inter[0]), int(inter[1])))
        if not local_cycles:
            continue
        shard_id = f"{video.video_id}__seg{idx:03d}"
        shards.append(
            DenseLabelShard(
                shard_id=shard_id,
                video_id=video.video_id,
                camera_id=video.camera_id,
                supervised_start_frame=int(interval[0]),
                supervised_end_frame=int(interval[1]),
                cycles=tuple(local_cycles),
                class_targets_rle=tuple(local_class_targets),
                class_label_map=dict(class_label_map),
                action_labels=tuple(action_label_rows),
                ignore_label=int(ignore_label_i),
                fallback_idle_label=int(fallback_idle_label_i),
            )
        )
    return shards
