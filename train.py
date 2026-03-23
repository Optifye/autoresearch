"""
Autoresearch dense-temporal training script. Single-GPU, single-file.

The shipped baseline is intentionally conservative:

- frozen V-JEPA Large encoder outputs
- current SSV2 pooler output (`pooled_z0`) by default
- TCN training over the prepared cache
- automatic task selection:
  - paired-boundary decoding for cyclic datasets
  - multiclass event classification for event datasets

Everything below the fixed `prepare.py` harness is fair game for the
autonomous agent to mutate.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.boundary_labels import (
    CycleInterval,
    build_boundary_targets,
    map_cycles_to_indices,
)
from autoresearch_vjepa.cache_contract import (
    CACHE_DIR,
    TIME_BUDGET,
    TOTAL_TIMEOUT_SECONDS,
    EventPair,
    SegmentRecord,
    decode_event_pairs,
    evaluate_predictions,
    load_manifest,
    load_split_records,
    load_segment_arrays,
    memmap_npz_member,
)
from autoresearch_vjepa.losses import focal_bce_with_logits, masked_mean
from autoresearch_vjepa.models.boundary_tcn import (
    BoundaryTCN,
    BoundaryTCNConfig,
)
from autoresearch_vjepa.pooler import build_probe_pooler


# ---------------------------------------------------------------------------
# Experiment config (the autonomous agent edits these directly)
# ---------------------------------------------------------------------------

MODEL_FAMILY = "tcn_then_probe_phase1"
AUX_MODE: Literal["none", "cycle_contrastive"] = "none"
TASK_MODE_OVERRIDE = ""

SEED = 42
GRAD_ACCUM_STEPS = 1
WARMUP_RATIO = 0.05
FINAL_LR_FRAC = 0.2

DEFAULT_TCN_STAGE_SECONDS = 180.0
DEFAULT_POOLER_STAGE_SECONDS = 420.0

TCN_BATCH_SIZE = 8
TCN_CHUNK_LEN = 256
TCN_LR = 1e-3
TCN_WEIGHT_DECAY = 1e-4

POOLER_BATCH_SIZE = 1
POOLER_CHUNK_LEN = 48
POOLER_LR = 1e-5
POOLER_WEIGHT_DECAY = 1e-4
POOLER_EVAL_TOKEN_CHUNK = 8

HIDDEN_DIM = 128
KERNEL_SIZE = 3
DROPOUT = 0.1
USE_LAYERNORM = True
DILATIONS = (1, 2, 4, 8, 16, 32)

IGNORE_RADIUS = 1
SMOOTH_SIGMA = 0.0
FOCAL_GAMMA = 2.0
POS_WEIGHT_START_END = 10.0
POS_WEIGHT_CYCLE = 1.0
CYCLE_LOSS_WEIGHT = 0.5
AUX_LOSS_WEIGHT = 0.1
NEG_MARGIN = 0.2
EVENT_CLASS_WEIGHT_POWER = 0.5
EVENT_LABEL_SMOOTHING = 0.0

EVAL_TOKEN_CHUNK = 64
LOG_EVERY = 20

TaskMode = Literal["boundary_pairs", "event_multiclass"]


def resolve_time_budget_seconds() -> float:
    raw = os.getenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "").strip()
    if not raw:
        return float(TIME_BUDGET)
    return max(1.0, float(raw))


def resolve_total_timeout_seconds() -> float:
    raw = os.getenv("AUTORESEARCH_TOTAL_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(TOTAL_TIMEOUT_SECONDS)
    return max(resolve_time_budget_seconds(), float(raw))


def resolve_stage_budget_seconds(total_budget_seconds: float) -> Tuple[float, float]:
    tcn_raw = os.getenv("AUTORESEARCH_TCN_STAGE_SECONDS", "").strip()
    pooler_raw = os.getenv("AUTORESEARCH_POOLER_STAGE_SECONDS", "").strip()
    if tcn_raw:
        tcn_seconds = max(0.0, float(tcn_raw))
    else:
        tcn_seconds = float(DEFAULT_TCN_STAGE_SECONDS)
    if pooler_raw:
        pooler_seconds = max(0.0, float(pooler_raw))
    else:
        pooler_seconds = float(DEFAULT_POOLER_STAGE_SECONDS)
    total_requested = float(tcn_seconds + pooler_seconds)
    if total_requested <= 0.0:
        raise RuntimeError("Stage budgets must sum to a positive duration.")
    scale = float(total_budget_seconds) / float(total_requested)
    return float(tcn_seconds * scale), float(pooler_seconds * scale)


@dataclass(frozen=True)
class SupervisedSegment:
    record: SegmentRecord
    task_mode: TaskMode
    local_start_idx: int
    local_end_idx: int
    global_start_idx: int
    global_end_idx: int
    timestamps_ms: np.ndarray
    pooled_z0: Optional[np.ndarray]
    y_start: Optional[np.ndarray]
    y_end: Optional[np.ndarray]
    y_cycle: Optional[np.ndarray]
    mask_start_end: Optional[np.ndarray]
    gt_pairs: Tuple[EventPair, ...]
    y_class: Optional[np.ndarray]
    mask_class: Optional[np.ndarray]

    @property
    def length(self) -> int:
        return int(self.timestamps_ms.shape[0])


@dataclass(frozen=True)
class TrainingStage:
    name: str
    representation_mode: Literal["pooled_z0", "tokens"]
    pooler_tune_mode: Literal["off", "full"]
    duration_seconds: float
    batch_size: int
    chunk_len: int
    lr: float
    weight_decay: float
    eval_token_chunk: int
    freeze_tcn: bool = False


@dataclass(frozen=True)
class EventClassSchema:
    label_ids: Tuple[int, ...]
    label_names: Tuple[str, ...]
    label_id_to_index: Dict[int, int]

    @property
    def num_classes(self) -> int:
        return int(len(self.label_ids))


class DenseTemporalModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        pooler_checkpoint: Optional[Path],
        representation_mode: str,
        pooler_tune_mode: str,
        device: torch.device,
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.representation_mode = str(representation_mode)
        self.pooler_tune_mode = str(pooler_tune_mode)
        self.device = device
        self.pooler: Optional[nn.Module] = None
        if self.representation_mode == "tokens":
            if pooler_checkpoint is None:
                raise RuntimeError("Token mode requires a readable pooler checkpoint.")
            self.pooler = build_probe_pooler(int(input_dim), device, Path(pooler_checkpoint))
            for param in self.pooler.parameters():
                param.requires_grad = self.pooler_tune_mode == "full"
        self.tcn = BoundaryTCN(
            BoundaryTCNConfig(
                input_dim=int(input_dim),
                hidden_dim=int(HIDDEN_DIM),
                out_dim=int(output_dim),
                kernel_size=int(KERNEL_SIZE),
                dropout=float(DROPOUT),
                use_layernorm=bool(USE_LAYERNORM),
                dilations=tuple(int(item) for item in DILATIONS),
            )
        )

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.pooler is None:
            raise RuntimeError("Pooler not initialised for token mode.")
        bsz, steps, num_tokens, dim = tokens.shape
        flat = tokens.reshape(bsz * steps, num_tokens, dim)
        pooled = self.pooler(flat).squeeze(1)
        return pooled.reshape(bsz, steps, dim)

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        if self.representation_mode == "tokens":
            return self._pool_tokens(batch)
        return batch

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(batch)
        logits = self.tcn(features)
        return logits, features


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(int(param.numel()) for param in module.parameters() if param.requires_grad))


@lru_cache(maxsize=4096)
def load_label_payload(label_path: str) -> dict:
    payload = json.loads(Path(label_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Label payload at {label_path} is not a JSON object.")
    return payload


@lru_cache(maxsize=512)
def load_resolved_config(source_run_dir: str) -> dict:
    path = Path(source_run_dir) / "resolved_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_action_label_name(value: object) -> str:
    text = str(value or "").strip()
    return text or "unknown"


def _normalize_label_map(raw_label_map: object) -> Dict[str, int]:
    if not isinstance(raw_label_map, dict):
        return {}
    out: Dict[str, int] = {}
    for key, value in raw_label_map.items():
        name = str(key or "").strip()
        if not name:
            continue
        try:
            out[name] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def _parse_action_label_rows(raw_action_labels: object, *, label_map: Dict[str, int]) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    seen_ids: set[int] = set()

    if isinstance(raw_action_labels, list):
        for item in raw_action_labels:
            if not isinstance(item, dict):
                continue
            try:
                label_id = int(item.get("label_id"))
            except (TypeError, ValueError):
                continue
            if label_id in seen_ids:
                continue
            seen_ids.add(label_id)
            rows.append((label_id, _normalize_action_label_name(item.get("label_name"))))

    if rows:
        return rows

    for name, label_id in label_map.items():
        if name in {"idle", "ignore", "action"}:
            continue
        label_id_i = int(label_id)
        if label_id_i in seen_ids:
            continue
        seen_ids.add(label_id_i)
        rows.append((label_id_i, _normalize_action_label_name(name)))
    return rows


def _nearest_index_ms(timestamps_ms: np.ndarray, target_ms: int) -> int:
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


def _map_ms_to_index(
    *,
    timestamps_ms: np.ndarray,
    value_ms: int,
    boundary_index_mode: str,
    is_end: bool,
) -> int:
    mode = str(boundary_index_mode or "nearest").strip().lower().replace("-", "_")
    if mode == "ordered_nearest":
        mode = "nearest"
    if mode == "legacy":
        if is_end:
            idx = int(np.searchsorted(timestamps_ms, int(value_ms), side="right") - 1)
        else:
            idx = int(np.searchsorted(timestamps_ms, int(value_ms), side="left"))
    else:
        idx = int(_nearest_index_ms(timestamps_ms, int(value_ms)))
    return int(max(0, min(idx, int(timestamps_ms.shape[0]) - 1)))


def _parse_class_target_spans(raw_spans: object) -> List[Tuple[int, int, int]]:
    spans: List[Tuple[int, int, int]] = []
    if not isinstance(raw_spans, list):
        return spans
    for item in raw_spans:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                label_id = int(item[0])
                start_frame = int(item[1])
                end_frame = int(item[2])
            except (TypeError, ValueError):
                continue
        elif isinstance(item, dict):
            try:
                label_id = int(item.get("label_id"))
                start_frame = int(item.get("start_frame"))
                end_frame = int(item.get("end_frame"))
            except (TypeError, ValueError):
                continue
        else:
            continue
        if end_frame < start_frame:
            continue
        spans.append((label_id, start_frame, end_frame))
    return spans


def resolve_task_mode(records: Sequence[SegmentRecord]) -> TaskMode:
    override = str(os.getenv("AUTORESEARCH_TASK_MODE", TASK_MODE_OVERRIDE)).strip().lower()
    if override in {"event", "event_multiclass"}:
        return "event_multiclass"
    if override in {"boundary", "boundary_pairs", "pairs", "cyclic"}:
        return "boundary_pairs"

    for record in records:
        resolved_cfg = load_resolved_config(record.source_run_dir)
        temporal_mode = str(resolved_cfg.get("temporal_structure_mode") or "").strip().lower()
        if temporal_mode == "event":
            return "event_multiclass"
        if temporal_mode:
            return "boundary_pairs"
        labels_payload = load_label_payload(record.label_path)
        label_map = _normalize_label_map(labels_payload.get("label_map"))
        action_rows = _parse_action_label_rows(labels_payload.get("action_labels"), label_map=label_map)
        if len(action_rows) >= 2 and _parse_class_target_spans(labels_payload.get("class_targets_rle")):
            return "event_multiclass"
    return "boundary_pairs"


def resolve_event_class_schema(records: Sequence[SegmentRecord]) -> EventClassSchema:
    ordered_label_ids: List[int] = []
    label_names: Dict[int, str] = {}

    for record in records:
        labels_payload = load_label_payload(record.label_path)
        label_map = _normalize_label_map(labels_payload.get("label_map"))
        rows = _parse_action_label_rows(labels_payload.get("action_labels"), label_map=label_map)
        if not rows:
            ignore_label = int(labels_payload.get("ignore_label", label_map.get("ignore", -1)))
            for label_id, _start_frame, _end_frame in _parse_class_target_spans(labels_payload.get("class_targets_rle")):
                if int(label_id) == int(ignore_label):
                    continue
                if int(label_id) not in label_names:
                    ordered_label_ids.append(int(label_id))
                    fallback_name = next((name for name, value in label_map.items() if int(value) == int(label_id)), None)
                    label_names[int(label_id)] = _normalize_action_label_name(fallback_name or f"class_{label_id}")
            continue
        for label_id, label_name in rows:
            label_id_i = int(label_id)
            if label_id_i not in label_names:
                ordered_label_ids.append(label_id_i)
                label_names[label_id_i] = _normalize_action_label_name(label_name)

    if not ordered_label_ids:
        raise RuntimeError("Event dataset detected, but no action-label schema could be resolved from label payloads.")

    return EventClassSchema(
        label_ids=tuple(ordered_label_ids),
        label_names=tuple(label_names[label_id] for label_id in ordered_label_ids),
        label_id_to_index={int(label_id): int(idx) for idx, label_id in enumerate(ordered_label_ids)},
    )


def build_event_targets(
    *,
    record: SegmentRecord,
    timestamps_ms: np.ndarray,
    event_schema: EventClassSchema,
) -> Tuple[np.ndarray, np.ndarray]:
    labels_payload = load_label_payload(record.label_path)
    label_map = _normalize_label_map(labels_payload.get("label_map"))
    ignore_label = int(labels_payload.get("ignore_label", label_map.get("ignore", -1)))
    fps = float(labels_payload.get("fps") or record.fps or 0.0)
    if fps <= 0.0:
        raise RuntimeError(f"Invalid fps for {record.segment_id}: {fps!r}")

    y_class = np.full((int(timestamps_ms.shape[0]),), -1, dtype=np.int64)
    mask_class = np.zeros((int(timestamps_ms.shape[0]),), dtype=np.float32)
    for label_id, start_frame, end_frame in _parse_class_target_spans(labels_payload.get("class_targets_rle")):
        label_id_i = int(label_id)
        if label_id_i == int(ignore_label):
            continue
        class_index = event_schema.label_id_to_index.get(label_id_i)
        if class_index is None:
            continue
        start_ms = int(round(float(start_frame) * 1000.0 / float(fps)))
        end_ms = int(round(float(end_frame) * 1000.0 / float(fps)))
        start_idx = _map_ms_to_index(
            timestamps_ms=timestamps_ms,
            value_ms=int(start_ms),
            boundary_index_mode="nearest",
            is_end=False,
        )
        end_idx = _map_ms_to_index(
            timestamps_ms=timestamps_ms,
            value_ms=int(end_ms),
            boundary_index_mode="nearest",
            is_end=True,
        )
        if end_idx < start_idx:
            continue
        y_class[start_idx : end_idx + 1] = int(class_index)
        mask_class[start_idx : end_idx + 1] = 1.0

    if float(mask_class.sum()) <= 0.0:
        raise RuntimeError(f"No event-class supervision mapped for segment {record.segment_id}")
    return y_class, mask_class


def compute_event_class_weights(
    segments: Sequence[SupervisedSegment],
    *,
    num_classes: int,
) -> Optional[torch.Tensor]:
    counts = np.zeros((int(num_classes),), dtype=np.float64)
    for segment in segments:
        if segment.y_class is None or segment.mask_class is None:
            continue
        valid = segment.mask_class > 0.5
        if not np.any(valid):
            continue
        counts += np.bincount(segment.y_class[valid], minlength=int(num_classes)).astype(np.float64, copy=False)
    positive = counts > 0
    if not np.any(positive):
        return None
    weights = np.ones_like(counts, dtype=np.float64)
    weights[positive] = 1.0 / np.power(counts[positive], float(EVENT_CLASS_WEIGHT_POWER))
    weights = weights / max(1e-8, float(weights.mean()))
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


def build_supervised_segment(
    record: SegmentRecord,
    *,
    use_eval_span: bool,
    preload_pooled: bool,
    task_mode: TaskMode,
    event_schema: Optional[EventClassSchema],
) -> SupervisedSegment:
    payload = load_segment_arrays(
        record,
        representation="pooled_z0" if preload_pooled else "pooled_z0",
        use_eval_span=use_eval_span,
    )
    timestamps_ms = payload["timestamps_ms"].astype(np.int64, copy=False)
    local_start_idx = 0
    local_end_idx = int(timestamps_ms.shape[0]) - 1
    global_start_idx = int(payload["slice_start_idx"])
    global_end_idx = int(payload["slice_end_idx"])
    pooled_z0 = payload["pooled_z0"].astype(np.float32, copy=False) if preload_pooled else None
    if str(task_mode) == "event_multiclass":
        if event_schema is None:
            raise RuntimeError("Event task mode requires an event class schema.")
        y_class, mask_class = build_event_targets(
            record=record,
            timestamps_ms=timestamps_ms,
            event_schema=event_schema,
        )
        return SupervisedSegment(
            record=record,
            task_mode=str(task_mode),
            local_start_idx=int(local_start_idx),
            local_end_idx=int(local_end_idx),
            global_start_idx=int(global_start_idx),
            global_end_idx=int(global_end_idx),
            timestamps_ms=timestamps_ms,
            pooled_z0=pooled_z0,
            y_start=None,
            y_end=None,
            y_cycle=None,
            mask_start_end=None,
            gt_pairs=tuple(),
            y_class=y_class.astype(np.int64, copy=False),
            mask_class=mask_class.astype(np.float32, copy=False),
        )
    gt_pairs = tuple(
        EventPair(start_ms=int(start_ms), end_ms=int(end_ms))
        for start_ms, end_ms in record.event_pairs_ms
        if int(start_ms) >= int(timestamps_ms[0]) and int(end_ms) <= int(timestamps_ms[-1])
    )
    mapped, _stats = map_cycles_to_indices(
        [CycleInterval(start_ms=item.start_ms, end_ms=item.end_ms) for item in gt_pairs],
        timestamps_ms,
        boundary_index_mode="ordered_nearest",
    )
    y_start, y_end, y_cycle, mask_start_end = build_boundary_targets(
        int(timestamps_ms.shape[0]),
        mapped,
        ignore_radius=int(IGNORE_RADIUS),
        smooth_sigma=float(SMOOTH_SIGMA),
    )
    return SupervisedSegment(
        record=record,
        task_mode=str(task_mode),
        local_start_idx=int(local_start_idx),
        local_end_idx=int(local_end_idx),
        global_start_idx=int(global_start_idx),
        global_end_idx=int(global_end_idx),
        timestamps_ms=timestamps_ms,
        pooled_z0=pooled_z0,
        y_start=y_start.astype(np.float32, copy=False),
        y_end=y_end.astype(np.float32, copy=False),
        y_cycle=y_cycle.astype(np.float32, copy=False),
        mask_start_end=mask_start_end.astype(np.float32, copy=False),
        gt_pairs=gt_pairs,
        y_class=None,
        mask_class=None,
    )


def build_supervised_segments(
    records: Sequence[SegmentRecord],
    *,
    use_eval_span: bool,
    preload_pooled: bool,
    task_mode: TaskMode,
    event_schema: Optional[EventClassSchema],
) -> List[SupervisedSegment]:
    segments: List[SupervisedSegment] = []
    skipped = 0
    for record in records:
        try:
            segment = build_supervised_segment(
                record,
                use_eval_span=use_eval_span,
                preload_pooled=preload_pooled,
                task_mode=task_mode,
                event_schema=event_schema,
            )
        except RuntimeError as exc:
            print(f"[skip] {record.segment_id} | {exc}")
            skipped += 1
            continue
        segments.append(segment)
    if not segments:
        raise RuntimeError("No supervised segments remained after label mapping.")
    if skipped > 0:
        print(f"[data] skipped_segments={skipped}")
    return segments


def load_token_chunk(segment: SupervisedSegment, local_start: int, local_end: int) -> np.ndarray:
    global_start = int(segment.global_start_idx + local_start)
    global_end = int(segment.global_start_idx + local_end)
    tokens = memmap_npz_member(Path(segment.record.feature_path), "tokens")
    return np.array(tokens[global_start : global_end + 1], dtype=np.float16, copy=True)


def _sample_chunk_bounds(length: int, *, chunk_len: int) -> Tuple[int, int]:
    if length <= int(chunk_len):
        return 0, int(length) - 1
    start = random.randint(0, int(length) - int(chunk_len))
    end = start + int(chunk_len) - 1
    return int(start), int(end)


def collate_train_batch(
    segments: Sequence[SupervisedSegment],
    *,
    task_mode: TaskMode,
    representation_mode: str,
    batch_size: int,
    chunk_len: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    chosen: List[Tuple[SupervisedSegment, int, int]] = []
    max_len = 0
    for _ in range(int(batch_size)):
        seg = random.choice(list(segments))
        lo, hi = _sample_chunk_bounds(seg.length, chunk_len=int(chunk_len))
        chosen.append((seg, lo, hi))
        max_len = max(max_len, int(hi - lo + 1))

    if str(representation_mode) == "tokens":
        feature_batch = torch.zeros((len(chosen), max_len, segments[0].record.tokens_per_window, segments[0].record.token_dim), dtype=torch.float16)
    else:
        feature_batch = torch.zeros((len(chosen), max_len, segments[0].record.embedding_dim), dtype=torch.float32)
    valid_mask = torch.zeros((len(chosen), max_len), dtype=torch.float32)
    if str(task_mode) == "event_multiclass":
        y_class = torch.full((len(chosen), max_len), -1, dtype=torch.int64)
        mask_class = torch.zeros((len(chosen), max_len), dtype=torch.float32)
    else:
        y_start = torch.zeros((len(chosen), max_len), dtype=torch.float32)
        y_end = torch.zeros((len(chosen), max_len), dtype=torch.float32)
        y_cycle = torch.zeros((len(chosen), max_len), dtype=torch.float32)
        mask_start_end = torch.zeros((len(chosen), max_len), dtype=torch.float32)

    for row, (seg, lo, hi) in enumerate(chosen):
        size = int(hi - lo + 1)
        if str(representation_mode) == "tokens":
            feature_batch[row, :size] = torch.from_numpy(load_token_chunk(seg, lo, hi))
        else:
            assert seg.pooled_z0 is not None
            feature_batch[row, :size] = torch.from_numpy(seg.pooled_z0[lo : hi + 1])
        if str(task_mode) == "event_multiclass":
            assert seg.y_class is not None and seg.mask_class is not None
            y_class[row, :size] = torch.from_numpy(seg.y_class[lo : hi + 1].astype(np.int64, copy=False))
            mask_class[row, :size] = torch.from_numpy(seg.mask_class[lo : hi + 1])
        else:
            assert seg.y_start is not None and seg.y_end is not None and seg.y_cycle is not None
            assert seg.mask_start_end is not None
            y_start[row, :size] = torch.from_numpy(seg.y_start[lo : hi + 1])
            y_end[row, :size] = torch.from_numpy(seg.y_end[lo : hi + 1])
            y_cycle[row, :size] = torch.from_numpy(seg.y_cycle[lo : hi + 1])
            mask_start_end[row, :size] = torch.from_numpy(seg.mask_start_end[lo : hi + 1])
        valid_mask[row, :size] = 1.0

    if str(task_mode) == "event_multiclass":
        targets = {
            "y_class": y_class,
            "mask_class": mask_class,
            "valid_mask": valid_mask,
        }
    else:
        targets = {
            "y_start": y_start,
            "y_end": y_end,
            "y_cycle": y_cycle,
            "mask_start_end": mask_start_end,
            "valid_mask": valid_mask,
        }
    return feature_batch, targets


def get_lr_multiplier(progress: float) -> float:
    warmup = float(WARMUP_RATIO)
    if progress <= 0.0:
        return 0.0 if warmup > 0.0 else 1.0
    if warmup > 0.0 and progress < warmup:
        return progress / warmup
    tail = float(FINAL_LR_FRAC)
    decay_progress = (progress - warmup) / max(1e-6, 1.0 - warmup)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return 1.0 - (1.0 - tail) * decay_progress


def compute_aux_loss(features: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    if AUX_MODE != "cycle_contrastive":
        return features.new_tensor(0.0)
    valid_mask = targets["valid_mask"]
    cycle_mask = targets["y_cycle"]
    feats = F.normalize(features, dim=-1, eps=1e-6)
    same_mask = valid_mask[:, 1:] * valid_mask[:, :-1] * cycle_mask[:, 1:] * cycle_mask[:, :-1]
    diff_mask = valid_mask[:, 1:] * valid_mask[:, :-1] * torch.abs(cycle_mask[:, 1:] - cycle_mask[:, :-1])
    cosine = (feats[:, 1:] * feats[:, :-1]).sum(dim=-1)
    same_loss = masked_mean(1.0 - cosine, same_mask)
    diff_loss = masked_mean(F.relu(cosine - float(NEG_MARGIN)), diff_mask)
    return same_loss + diff_loss


def compute_event_loss(
    logits: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    *,
    class_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits_2d = logits.reshape(-1, int(logits.shape[-1]))
    mask_1d = (targets["mask_class"] * targets["valid_mask"]).reshape(-1).to(device=logits.device, dtype=logits.dtype)
    target_1d = targets["y_class"].reshape(-1).to(device=logits.device, dtype=torch.long)
    safe_targets = torch.where(mask_1d > 0.0, target_1d, torch.zeros_like(target_1d))
    weight = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    ce = F.cross_entropy(
        logits_2d,
        safe_targets,
        weight=weight,
        reduction="none",
        label_smoothing=float(EVENT_LABEL_SMOOTHING),
    )
    class_loss = masked_mean(ce, mask_1d)
    stats = {
        "loss_total": float(class_loss.detach().item()),
        "loss_class": float(class_loss.detach().item()),
        "loss_aux": 0.0,
    }
    return class_loss, stats


def compute_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    *,
    task_mode: TaskMode,
    class_weights: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if str(task_mode) == "event_multiclass":
        del features
        return compute_event_loss(logits, targets, class_weights=class_weights)

    start_logits = logits[:, :, 0]
    end_logits = logits[:, :, 1]
    cycle_logits = logits[:, :, 2]

    pos_weight_start_end = torch.tensor(float(POS_WEIGHT_START_END), device=logits.device)
    pos_weight_cycle = torch.tensor(float(POS_WEIGHT_CYCLE), device=logits.device)

    start_loss_raw = focal_bce_with_logits(
        start_logits,
        targets["y_start"].to(logits.device),
        pos_weight=pos_weight_start_end,
        gamma=float(FOCAL_GAMMA),
    )
    end_loss_raw = focal_bce_with_logits(
        end_logits,
        targets["y_end"].to(logits.device),
        pos_weight=pos_weight_start_end,
        gamma=float(FOCAL_GAMMA),
    )
    cycle_loss_raw = focal_bce_with_logits(
        cycle_logits,
        targets["y_cycle"].to(logits.device),
        pos_weight=pos_weight_cycle,
        gamma=float(FOCAL_GAMMA),
    )

    mask_start_end = targets["mask_start_end"].to(logits.device) * targets["valid_mask"].to(logits.device)
    valid_mask = targets["valid_mask"].to(logits.device)

    start_loss = masked_mean(start_loss_raw, mask_start_end)
    end_loss = masked_mean(end_loss_raw, mask_start_end)
    cycle_loss = masked_mean(cycle_loss_raw, valid_mask)
    aux_loss = compute_aux_loss(features, {k: v.to(logits.device) for k, v in targets.items()})

    total = start_loss + end_loss + (float(CYCLE_LOSS_WEIGHT) * cycle_loss) + (float(AUX_LOSS_WEIGHT) * aux_loss)
    stats = {
        "loss_total": float(total.detach().item()),
        "loss_start": float(start_loss.detach().item()),
        "loss_end": float(end_loss.detach().item()),
        "loss_cycle": float(cycle_loss.detach().item()),
        "loss_aux": float(aux_loss.detach().item()),
    }
    return total, stats


def _forward_segment_logits(
    model: DenseTemporalModel,
    segment: SupervisedSegment,
    device: torch.device,
    *,
    eval_token_chunk: int,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        if str(model.representation_mode) == "tokens":
            for lo in range(0, segment.length, int(eval_token_chunk)):
                hi = min(segment.length, lo + int(eval_token_chunk))
                tokens = load_token_chunk(segment, lo, hi - 1)
                batch = torch.from_numpy(tokens).unsqueeze(0).to(device)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with autocast_ctx:
                    logits, _ = model(batch)
                outputs.append(logits.squeeze(0).detach().float().cpu().numpy().astype(np.float32, copy=False))
        else:
            assert segment.pooled_z0 is not None
            batch = torch.from_numpy(segment.pooled_z0).unsqueeze(0).to(device)
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with autocast_ctx:
                logits, _ = model(batch)
            outputs.append(logits.squeeze(0).detach().float().cpu().numpy().astype(np.float32, copy=False))
    if outputs:
        return np.concatenate(outputs, axis=0)
    return np.zeros((0, int(model.tcn.cfg.out_dim)), dtype=np.float32)


def evaluate_boundary_model(
    model: DenseTemporalModel,
    val_segments: Sequence[SupervisedSegment],
    device: torch.device,
    *,
    eval_token_chunk: int,
) -> Dict[str, float]:
    predictions: Dict[str, Sequence[EventPair]] = {}
    for segment in val_segments:
        logits = _forward_segment_logits(model, segment, device, eval_token_chunk=int(eval_token_chunk))
        probs = 1.0 / (1.0 + np.exp(-logits))
        predictions[segment.record.segment_id] = decode_event_pairs(probs, segment.timestamps_ms)
    return evaluate_predictions(predictions, records=[segment.record for segment in val_segments])


def _safe_metric_name(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def evaluate_event_model(
    model: DenseTemporalModel,
    val_segments: Sequence[SupervisedSegment],
    device: torch.device,
    *,
    eval_token_chunk: int,
    event_schema: EventClassSchema,
) -> Dict[str, float]:
    confusion = np.zeros((int(event_schema.num_classes), int(event_schema.num_classes)), dtype=np.int64)
    for segment in val_segments:
        logits = _forward_segment_logits(model, segment, device, eval_token_chunk=int(eval_token_chunk))
        pred = np.asarray(np.argmax(logits, axis=-1), dtype=np.int64)
        assert segment.y_class is not None and segment.mask_class is not None
        mask = segment.mask_class > 0.5
        if pred.shape[0] != segment.y_class.shape[0]:
            raise RuntimeError(
                f"Prediction length mismatch for {segment.record.segment_id}: pred={pred.shape[0]} target={segment.y_class.shape[0]}"
            )
        true = segment.y_class[mask]
        pred_valid = pred[mask]
        for true_idx, pred_idx in zip(true, pred_valid):
            if 0 <= int(true_idx) < int(event_schema.num_classes) and 0 <= int(pred_idx) < int(event_schema.num_classes):
                confusion[int(true_idx), int(pred_idx)] += 1

    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    accuracy = float(correct / max(1, total))
    per_class_f1: List[float] = []
    metrics: Dict[str, float] = {
        "val_accuracy": float(accuracy),
        "val_macro_f1": 0.0,
        "val_labeled_windows": float(total),
    }
    for class_idx, class_name in enumerate(event_schema.label_names):
        tp = float(confusion[class_idx, class_idx])
        fp = float(confusion[:, class_idx].sum() - tp)
        fn = float(confusion[class_idx, :].sum() - tp)
        support = float(confusion[class_idx, :].sum())
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
        if support > 0.0:
            per_class_f1.append(float(f1))
        suffix = _safe_metric_name(class_name)
        metrics[f"val_support_{suffix}"] = float(support)
        metrics[f"val_precision_{suffix}"] = float(precision)
        metrics[f"val_recall_{suffix}"] = float(recall)
        metrics[f"val_f1_{suffix}"] = float(f1)
    metrics["val_macro_f1"] = float(np.mean(per_class_f1)) if per_class_f1 else 0.0
    return metrics


def find_default_pooler(records: Sequence[SegmentRecord]) -> Optional[Path]:
    for record in records:
        raw = str(record.pooler_checkpoint or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path
    return None


def _workspace_root_candidates() -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in (
        os.getenv("AUTORESEARCH_WORKSPACE_ROOT", "").strip(),
        "/workspace",
        str(ROOT.parent),
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        resolved = path.resolve() if path.exists() else path
        key = str(resolved)
        if key in seen:
            continue
        out.append(resolved)
        seen.add(key)
    return out


def _normalize_workspace_path(raw: str) -> Optional[Path]:
    raw_n = str(raw or "").strip()
    if not raw_n:
        return None
    direct = Path(raw_n).expanduser()
    if direct.exists():
        return direct.resolve()
    workspace_prefix = "/workspace/"
    if raw_n.startswith(workspace_prefix):
        rel = raw_n[len(workspace_prefix) :]
        for root in _workspace_root_candidates():
            candidate = (root / rel).resolve() if (root / rel).exists() else (root / rel)
            if candidate.exists():
                return candidate.resolve()
    return None


def _infer_encoder_model_from_pooler_path(pooler_checkpoint: Path) -> str:
    name = pooler_checkpoint.name.lower()
    if "vitg" in name or "giant" in name:
        return "giant"
    return "large"


def _default_encoder_checkpoint_for_model(encoder_model: str) -> Optional[Path]:
    model_n = str(encoder_model or "large").strip().lower()
    filenames = ["vitl.pt"] if model_n == "large" else ["vitg-384.pt", "vitl.pt"]
    for root in _workspace_root_candidates():
        for filename in filenames:
            candidate = root / "encoder_models" / filename
            if candidate.exists():
                return candidate.resolve()
    return None


def _encoder_checkpoint_from_pooler_path(pooler_checkpoint: Path, encoder_model: str) -> Optional[Path]:
    model_n = str(encoder_model or "large").strip().lower()
    filenames = ["vitl.pt"] if model_n == "large" else ["vitg-384.pt", "vitl.pt"]
    pooler_dir = Path(pooler_checkpoint).expanduser().resolve().parent
    encoder_models_root = pooler_dir.parent if pooler_dir.name == "vjepa2_attention_poolers" else pooler_dir
    for filename in filenames:
        candidate = encoder_models_root / filename
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_stage_encoder_settings(
    records: Sequence[SegmentRecord],
    *,
    pooler_checkpoint: Path,
) -> Tuple[str, Path]:
    for record in records:
        resolved_cfg = Path(record.source_run_dir) / "resolved_config.json"
        if not resolved_cfg.exists():
            continue
        try:
            payload = json.loads(resolved_cfg.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_cfg = payload.get("model") if isinstance(payload, dict) else None
        if not isinstance(model_cfg, dict):
            model_cfg = payload.get("temporal_model") if isinstance(payload, dict) else None
        if not isinstance(model_cfg, dict):
            continue
        encoder_model = str(model_cfg.get("encoder_model") or "").strip().lower()
        encoder_checkpoint = _normalize_workspace_path(str(model_cfg.get("encoder_checkpoint") or ""))
        if encoder_model and encoder_checkpoint is not None and encoder_checkpoint.exists():
            return encoder_model, encoder_checkpoint

    encoder_model = _infer_encoder_model_from_pooler_path(Path(pooler_checkpoint))
    encoder_checkpoint = _encoder_checkpoint_from_pooler_path(Path(pooler_checkpoint), str(encoder_model))
    if encoder_checkpoint is None:
        encoder_checkpoint = _default_encoder_checkpoint_for_model(str(encoder_model))
    if encoder_checkpoint is None:
        raise RuntimeError(f"Unable to resolve encoder checkpoint for pooler {pooler_checkpoint}")
    return str(encoder_model), encoder_checkpoint


def configure_stage_pooler_env(*, encoder_model: str, encoder_checkpoint: Path) -> None:
    os.environ["VJEPA_ENCODER_MODEL"] = str(encoder_model)
    os.environ["VJEPA_ENCODER_CHECKPOINT"] = str(Path(encoder_checkpoint).resolve())
    os.environ["DENSE_TEMPORAL_ENCODER_MODEL"] = str(encoder_model)
    os.environ["DENSE_TEMPORAL_ENCODER_CHECKPOINT"] = str(Path(encoder_checkpoint).resolve())


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def build_stage_model(
    *,
    stage: TrainingStage,
    input_dim: int,
    output_dim: int,
    pooler_checkpoint: Optional[Path],
    device: torch.device,
    tcn_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    encoder_model_hint: Optional[str] = None,
    encoder_checkpoint_hint: Optional[Path] = None,
) -> DenseTemporalModel:
    if str(stage.representation_mode) == "tokens":
        if pooler_checkpoint is None:
            raise RuntimeError("Token-mode stage requires a readable pooler checkpoint.")
        if encoder_model_hint is None or encoder_checkpoint_hint is None:
            raise RuntimeError("Token-mode stage requires resolved encoder settings.")
        configure_stage_pooler_env(
            encoder_model=str(encoder_model_hint),
            encoder_checkpoint=Path(encoder_checkpoint_hint),
        )
    model = DenseTemporalModel(
        input_dim=int(input_dim),
        pooler_checkpoint=pooler_checkpoint if str(stage.representation_mode) == "tokens" else None,
        representation_mode=str(stage.representation_mode),
        pooler_tune_mode=str(stage.pooler_tune_mode),
        device=device,
        output_dim=int(output_dim),
    ).to(device)
    if tcn_state_dict is not None:
        model.tcn.load_state_dict(tcn_state_dict, strict=True)
    if bool(stage.freeze_tcn):
        for param in model.tcn.parameters():
            param.requires_grad = False
    return model


def build_stage_optimizer(
    model: DenseTemporalModel,
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for stage.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(lr), weight_decay=float(weight_decay))
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", float(group["lr"]))
    return optimizer


def run_training_stage(
    *,
    stage: TrainingStage,
    model: DenseTemporalModel,
    train_segments: Sequence[SupervisedSegment],
    task_mode: TaskMode,
    class_weights: Optional[torch.Tensor],
    device: torch.device,
    use_autocast: bool,
    autocast_dtype: torch.dtype,
    total_start_time: float,
    total_timeout_seconds: float,
    total_training_seconds_before: float,
    step_before: int,
    peak_vram_mb_before: float,
) -> Tuple[float, int, float]:
    optimizer = build_stage_optimizer(model, lr=float(stage.lr), weight_decay=float(stage.weight_decay))
    stage_start_time = time.time()
    stage_steps = 0
    smooth_loss = 0.0
    peak_vram_mb = float(peak_vram_mb_before)

    while True:
        if (time.time() - total_start_time) >= float(total_timeout_seconds):
            raise RuntimeError("FAIL: exceeded total timeout")

        optimizer.zero_grad(set_to_none=True)
        batch_t0 = time.time()
        if str(task_mode) == "event_multiclass":
            running_stats = {"loss_total": 0.0, "loss_class": 0.0, "loss_aux": 0.0}
        else:
            running_stats = {"loss_total": 0.0, "loss_start": 0.0, "loss_end": 0.0, "loss_cycle": 0.0, "loss_aux": 0.0}
        for _micro_step in range(int(GRAD_ACCUM_STEPS)):
            features_cpu, targets = collate_train_batch(
                train_segments,
                task_mode=task_mode,
                representation_mode=str(stage.representation_mode),
                batch_size=int(stage.batch_size),
                chunk_len=int(stage.chunk_len),
            )
            batch = features_cpu.to(device, non_blocking=True)
            targets = {key: value.to(device, non_blocking=True) for key, value in targets.items()}
            autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
            with autocast_ctx:
                logits, features = model(batch)
                loss, stats = compute_loss(
                    logits,
                    features,
                    targets,
                    task_mode=task_mode,
                    class_weights=class_weights,
                )
                loss = loss / float(GRAD_ACCUM_STEPS)
            loss.backward()
            for key, value in stats.items():
                running_stats[key] += float(value)

        torch.nn.utils.clip_grad_norm_([param for param in model.parameters() if param.requires_grad], max_norm=1.0)
        stage_training_seconds = time.time() - stage_start_time
        progress = min(max(stage_training_seconds / max(1e-6, float(stage.duration_seconds)), 0.0), 1.0)
        lr_mult = get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = float(group.get("initial_lr", stage.lr)) * float(lr_mult)
        optimizer.step()

        batch_dt = time.time() - batch_t0
        stage_steps += 1
        smooth_loss = 0.9 * smooth_loss + 0.1 * (running_stats["loss_total"] / float(max(1, GRAD_ACCUM_STEPS)))
        if device.type == "cuda":
            peak_vram_mb = max(peak_vram_mb, float(torch.cuda.max_memory_allocated(device)) / 1024.0 / 1024.0)

        total_steps = int(step_before + stage_steps)
        if total_steps % int(LOG_EVERY) == 0:
            stage_remaining = max(0.0, float(stage.duration_seconds) - float(stage_training_seconds))
            total_remaining = max(0.0, float(total_timeout_seconds) - float(time.time() - total_start_time))
            if str(task_mode) == "event_multiclass":
                print(
                    f"[{stage.name}] step {total_steps:05d} | loss={smooth_loss:.5f} | "
                    f"class={running_stats['loss_class']:.4f} aux={running_stats['loss_aux']:.4f} | "
                    f"dt={batch_dt*1000.0:.0f}ms | stage_remaining={stage_remaining:.0f}s "
                    f"| total_remaining={total_remaining:.0f}s"
                )
            else:
                print(
                    f"[{stage.name}] step {total_steps:05d} | loss={smooth_loss:.5f} | "
                    f"start={running_stats['loss_start']:.4f} end={running_stats['loss_end']:.4f} "
                    f"cycle={running_stats['loss_cycle']:.4f} aux={running_stats['loss_aux']:.4f} | "
                    f"dt={batch_dt*1000.0:.0f}ms | stage_remaining={stage_remaining:.0f}s "
                    f"| total_remaining={total_remaining:.0f}s"
                )

        if stage_training_seconds >= float(stage.duration_seconds):
            return stage_training_seconds, total_steps, peak_vram_mb


def main() -> int:
    manifest = load_manifest()
    set_seed(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_budget_seconds = resolve_time_budget_seconds()
    total_timeout_seconds = resolve_total_timeout_seconds()
    tcn_stage_seconds, pooler_stage_seconds = resolve_stage_budget_seconds(time_budget_seconds)
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_autocast else torch.float32

    train_records = load_split_records("train")
    if not train_records:
        raise RuntimeError("No train segments found. Run `python prepare.py` first.")
    task_mode = resolve_task_mode(train_records)
    event_schema = resolve_event_class_schema(train_records) if str(task_mode) == "event_multiclass" else None
    if str(task_mode) == "event_multiclass":
        val_records = load_split_records("val")
        if not val_records:
            raise RuntimeError("No val segments found. Adjust prepare filters or rebuild cache.")
        train_segments = build_supervised_segments(
            train_records,
            use_eval_span=False,
            preload_pooled=True,
            task_mode=task_mode,
            event_schema=event_schema,
        )
        val_segments = build_supervised_segments(
            val_records,
            use_eval_span=False,
            preload_pooled=True,
            task_mode=task_mode,
            event_schema=event_schema,
        )
        assert event_schema is not None
        output_dim = int(event_schema.num_classes)
        class_weights = compute_event_class_weights(train_segments, num_classes=int(event_schema.num_classes))
    else:
        val_records = load_split_records("val_eval")
        if not val_records:
            raise RuntimeError("No val_eval segments found. Adjust prepare filters or rebuild cache.")
        train_segments = build_supervised_segments(
            train_records,
            use_eval_span=False,
            preload_pooled=True,
            task_mode=task_mode,
            event_schema=None,
        )
        val_segments = build_supervised_segments(
            val_records,
            use_eval_span=True,
            preload_pooled=True,
            task_mode=task_mode,
            event_schema=None,
        )
        output_dim = 3
        class_weights = None

    input_dim = int(train_segments[0].record.embedding_dim)
    pooler_checkpoint = find_default_pooler(train_records)
    if pooler_checkpoint is None:
        raise RuntimeError("No readable pooler checkpoint found in prepared cache.")
    encoder_model, encoder_checkpoint = resolve_stage_encoder_settings(
        train_records,
        pooler_checkpoint=pooler_checkpoint,
    )
    if class_weights is not None:
        print(
            f"[task] {task_mode} | classes={event_schema.label_names if event_schema is not None else ()} | "
            f"class_weights={[round(float(item), 4) for item in class_weights.tolist()]}"
        )
    else:
        print(f"[task] {task_mode}")

    stages = [
        TrainingStage(
            name="tcn",
            representation_mode="pooled_z0",
            pooler_tune_mode="off",
            duration_seconds=float(tcn_stage_seconds),
            batch_size=int(TCN_BATCH_SIZE),
            chunk_len=int(TCN_CHUNK_LEN),
            lr=float(TCN_LR),
            weight_decay=float(TCN_WEIGHT_DECAY),
            eval_token_chunk=int(EVAL_TOKEN_CHUNK),
            freeze_tcn=False,
        ),
        TrainingStage(
            name="probe_phase1",
            representation_mode="tokens",
            pooler_tune_mode="full",
            duration_seconds=float(pooler_stage_seconds),
            batch_size=int(POOLER_BATCH_SIZE),
            chunk_len=int(POOLER_CHUNK_LEN),
            lr=float(POOLER_LR),
            weight_decay=float(POOLER_WEIGHT_DECAY),
            eval_token_chunk=int(POOLER_EVAL_TOKEN_CHUNK),
            freeze_tcn=True,
        ),
    ]

    t_start = time.time()
    total_training_time = 0.0
    total_steps = 0
    peak_vram_mb = 0.0
    final_model: Optional[DenseTemporalModel] = None
    final_stage: Optional[TrainingStage] = None
    tcn_state_dict: Optional[Dict[str, torch.Tensor]] = None
    stage_trainable_params: Dict[str, int] = {}

    for stage in stages:
        if float(stage.duration_seconds) <= 0.0:
            continue
        model = build_stage_model(
            stage=stage,
            input_dim=input_dim,
            output_dim=int(output_dim),
            pooler_checkpoint=pooler_checkpoint,
            device=device,
            tcn_state_dict=tcn_state_dict,
            encoder_model_hint=encoder_model,
            encoder_checkpoint_hint=encoder_checkpoint,
        )
        stage_trainable_params[stage.name] = count_trainable_params(model)
        print(
            f"[stage] {stage.name} | duration={stage.duration_seconds:.1f}s | "
            f"representation={stage.representation_mode} | pooler_tune={stage.pooler_tune_mode} | "
            f"trainable_params_M={stage_trainable_params[stage.name] / 1e6:.3f}"
        )
        try:
            stage_elapsed, total_steps, peak_vram_mb = run_training_stage(
                stage=stage,
                model=model,
                train_segments=train_segments,
                task_mode=task_mode,
                class_weights=class_weights,
                device=device,
                use_autocast=use_autocast,
                autocast_dtype=autocast_dtype,
                total_start_time=t_start,
                total_timeout_seconds=total_timeout_seconds,
                total_training_seconds_before=total_training_time,
                step_before=total_steps,
                peak_vram_mb_before=peak_vram_mb,
            )
        except RuntimeError as exc:
            print(str(exc))
            return 1
        total_training_time += float(stage_elapsed)
        tcn_state_dict = clone_state_dict(model.tcn)
        final_model = model
        final_stage = stage

    if final_model is None or final_stage is None:
        raise RuntimeError("No training stages executed.")

    if str(task_mode) == "event_multiclass":
        assert event_schema is not None
        metrics = evaluate_event_model(
            final_model,
            val_segments,
            device,
            eval_token_chunk=int(final_stage.eval_token_chunk),
            event_schema=event_schema,
        )
    else:
        metrics = evaluate_boundary_model(final_model, val_segments, device, eval_token_chunk=int(final_stage.eval_token_chunk))
    t_end = time.time()
    total_seconds = float(t_end - t_start)
    if total_seconds > float(total_timeout_seconds):
        print("FAIL: exceeded total timeout")
        return 1

    print("---")
    if str(task_mode) == "event_multiclass":
        print(f"val_accuracy:       {metrics['val_accuracy']:.6f}")
        print(f"val_macro_f1:       {metrics['val_macro_f1']:.6f}")
        print(f"val_labeled_windows:{metrics['val_labeled_windows']:.0f}")
        assert event_schema is not None
        for class_name in event_schema.label_names:
            suffix = _safe_metric_name(class_name)
            print(f"val_precision_{suffix}: {metrics[f'val_precision_{suffix}']:.6f}")
            print(f"val_recall_{suffix}:    {metrics[f'val_recall_{suffix}']:.6f}")
            print(f"val_f1_{suffix}:        {metrics[f'val_f1_{suffix}']:.6f}")
            print(f"val_support_{suffix}:   {metrics[f'val_support_{suffix}']:.0f}")
    else:
        print(f"val_pair_f1:        {metrics['val_pair_f1']:.6f}")
        print(f"val_pair_precision: {metrics['val_pair_precision']:.6f}")
        print(f"val_pair_recall:    {metrics['val_pair_recall']:.6f}")
        print(f"val_count_mae:      {metrics['val_count_mae']:.6f}")
        print(f"val_start_mae_ms:   {metrics['val_start_mae_ms']:.1f}")
        print(f"val_end_mae_ms:     {metrics['val_end_mae_ms']:.1f}")
    print(f"training_seconds:   {total_training_time:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"time_budget_seconds:{time_budget_seconds:.1f}")
    print(f"tcn_stage_seconds:  {tcn_stage_seconds:.1f}")
    print(f"pooler_stage_seconds:{pooler_stage_seconds:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_steps:          {total_steps}")
    print(f"tcn_trainable_params_M: {stage_trainable_params.get('tcn', 0) / 1e6:.3f}")
    print(f"pooler_trainable_params_M: {stage_trainable_params.get('probe_phase1', 0) / 1e6:.3f}")
    print(f"model_family:       {MODEL_FAMILY}")
    print(f"task_mode:          {task_mode}")
    print(f"pooler_tune_mode:   {final_stage.pooler_tune_mode}")
    print(f"representation_mode:{final_stage.representation_mode}")
    print(f"cache_root:         {CACHE_DIR}")
    print(f"train_segments:     {len(train_segments)}")
    print(f"val_segments:       {len(val_segments)}")
    print(f"cache_version:      {manifest.get('cache_version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
