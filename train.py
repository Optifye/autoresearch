"""
Replay-faithful autoresearch trainer for the Minda subassembly cyclic dataset.

This branch intentionally focuses on one dataset and one recipe:

- latest `minda-subassembly-tcn` dense-temporal source run
- hybrid halo16 bidirectional TCN stage-0 recipe
- historical probe phase-1 path kept intact but defaulted to zero budget
- deterministic 60/40 camera-stratified validation split
- fixed 5-minute pure TCN budget by default

The goal here is not to redesign the dense-temporal stack. It is to make the
validated recipe easy to rerun inside a single standalone `train.py`.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.cache_contract import (
    CACHE_VERSION,
    DEFAULT_SPLIT_POLICY,
    TIME_BUDGET,
    TOTAL_TIMEOUT_SECONDS,
    VAL_RATIO,
    EventPair,
    SegmentRecord,
    build_cache,
    configure_cache_paths,
    decode_event_pairs,
    evaluate_predictions,
    load_manifest,
    load_segment_arrays,
    load_split_records,
)


def _resolve_workspace_root() -> Path:
    candidates = []
    raw_env = os.getenv("AUTORESEARCH_WORKSPACE_ROOT", "").strip()
    if raw_env:
        candidates.append(Path(raw_env).expanduser())
    candidates.extend(
        [
            ROOT.parent / "internvideo-attention",
            Path("/home/ubuntu/internvideo-attention"),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Unable to resolve AUTORESEARCH_WORKSPACE_ROOT. "
        "Set AUTORESEARCH_WORKSPACE_ROOT to the internvideo-attention repo."
    )


WORKSPACE_ROOT = _resolve_workspace_root()
for candidate in (
    WORKSPACE_ROOT,
    WORKSPACE_ROOT / "src",
    WORKSPACE_ROOT / "clustering_exp",
):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from src.training.dense_temporal.boundary_labels import CycleInterval, build_boundary_targets, map_cycles_to_indices
from src.training.dense_temporal.losses import focal_bce_with_logits, masked_mean
from src.training.dense_temporal.models.boundary_tcn import BoundaryTCN, BoundaryTCNConfig
from src.training.dense_temporal import probe_phase1 as prod_probe
from src.training.dense_temporal import tcn_train as prod_tcn

LOGGER = logging.getLogger("autoresearch.subassembly.train")

MODEL_FAMILY = "minda_subassembly_hybrid_halo16_stage0_optional_historical_probe"
TASK_MODE = "boundary_pairs_single_space"
SEED = 42

SPACE_NAME = "minda-subassembly-tcn"
DEFAULT_SOURCE_RUN_DIR = Path("/tmp/embedding_runs/92c8fdb4-c0f6-4503-b2cc-ab340f79f8f6/dense_temporal")
LEGACY_SOURCE_RUN_DIR = Path("/tmp/autoresearch_minda_sources/minda-subassembly-tcn/dense_temporal")
DEFAULT_CACHE_DIR = Path("/tmp/autoresearch-minda-subassembly-cache")
DEFAULT_OUTPUT_ROOT = Path("/tmp/autoresearch_minda_subassembly_replay_runs")

DEFAULT_TCN_STAGE_SECONDS = 300.0
DEFAULT_PROBE_STAGE_SECONDS = 0.0
EVAL_TOKEN_CHUNK = 64
PHASE1_MIN_CHUNK = 8
VAL_PROXY_THRESHOLD_PROB = 0.15
VAL_PROXY_TOLERANCE_WINDOWS = 2
HALO_CONTEXT_WINDOWS = 16
HALO_CENTER_CHUNK = 256
STATE_IDLE = 0
STATE_START = 1
STATE_END = 2
STATE_WORK = 3
STATE_NAMES = ("idle", "start", "end", "work")
STATE_AUX_WEIGHT = 0.1
STATE_SHOULDER_RADIUS = 0
STATE_SHOULDER_WEIGHT = 0.5
STAGE0_MATERIAL_DEGRADE_EPS = 0.01


@dataclass(frozen=True)
class SplitPlan:
    split_policy: str
    train_records: Tuple[SegmentRecord, ...]
    val_records: Tuple[SegmentRecord, ...]
    val_eval_records: Tuple[SegmentRecord, ...]
    train_videos: Tuple[str, ...]
    val_videos: Tuple[str, ...]
    camera_val_counts: Dict[str, int]
    camera_total_counts: Dict[str, int]
    target_val_videos: float


@dataclass(frozen=True)
class SpaceSpec:
    name: str
    source_run_dir: Path
    cache_root: Path
    output_dir: Path
    split_plan: SplitPlan
    pooler_path: Path
    encoder_model: str
    encoder_checkpoint: Path


@dataclass(frozen=True)
class Stage0Config:
    seed: int = 42
    epochs: int = 200
    val_every_epochs: int = 5
    batch_size: int = 32
    chunk_len: int = 256
    grad_accum_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.05
    final_lr_frac: float = 0.2
    hidden_dim: int = 256
    kernel_size: int = 5
    dropout: float = 0.1
    use_layernorm: bool = True
    dilations: Tuple[int, ...] = (1, 2, 4, 8, 1, 2, 4, 8)
    focal_gamma: float = 2.0
    pos_weight_start_end: float = 10.0
    pos_weight_cycle: float = 1.0
    cycle_loss_weight: float = 0.5
    aux_loss_weight: float = 0.1
    transition_consistency_weight: float = 0.15
    neg_margin: float = 0.2
    ignore_radius: int = 1
    smooth_sigma: float = 0.0
    ema_enabled: bool = True
    ema_decay: float = 0.9998
    ema_start_ratio: float = 0.15
    boundary_index_mode: str = "ordered_nearest"
    model_family: str = MODEL_FAMILY
    seq_model: str = "tcn"
    combine_start_end: bool = False
    grad_clip_norm: float = 1.0
    task_specific_heads: bool = True
    bidirectional: bool = True
    log_every_steps: int = 20


@dataclass(frozen=True)
class SupervisedSegment:
    record: SegmentRecord
    global_start_idx: int
    global_end_idx: int
    timestamps_ms: np.ndarray
    y_start: np.ndarray
    y_end: np.ndarray
    y_cycle: np.ndarray
    mask_start_end: np.ndarray
    pooled_z0: np.ndarray
    y_state: np.ndarray
    state_weight: np.ndarray
    mapped_cycles_idx: Tuple[Tuple[int, int], ...]
    gt_pairs: Tuple[EventPair, ...]

    @property
    def length(self) -> int:
        return int(self.timestamps_ms.shape[0])


@dataclass(frozen=True)
class ValidationEvalSegment:
    record: SegmentRecord
    timestamps_ms: np.ndarray
    logits: np.ndarray
    mapped_cycles_idx: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class Stage0Result:
    checkpoint_path: Path
    config_snapshot_path: Path
    metrics_path: Path
    history_path: Path
    metrics: Dict[str, float]
    halo_metrics: Dict[str, float]
    fullseq_metrics: Dict[str, float]
    proxy_metrics: Dict[str, float]
    best_epoch: int
    best_source: str
    steps_per_epoch: int
    total_steps: int
    elapsed_seconds: float


@dataclass(frozen=True)
class Phase1Result:
    probe_checkpoint: Path
    tcn_checkpoint: Path
    metrics_path: Path
    config_snapshot_path: Path
    history_path: Path
    metrics: Dict[str, float]
    best_epoch: int
    best_loss: float
    effective_chunk_len: int
    elapsed_seconds: float
    skipped: bool = False


class Stage0BoundaryModel(nn.Module):
    def __init__(self, *, input_dim: int, cfg: Stage0Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.tcn = BoundaryTCN(
            BoundaryTCNConfig(
                input_dim=int(input_dim),
                hidden_dim=int(cfg.hidden_dim),
                out_dim=3,
                kernel_size=int(cfg.kernel_size),
                dropout=float(cfg.dropout),
                use_layernorm=bool(cfg.use_layernorm),
                dilations=tuple(int(item) for item in cfg.dilations),
                bidirectional=bool(cfg.bidirectional),
                task_specific_heads=bool(cfg.task_specific_heads),
                base_heads=3,
            )
        )
        self.state_head = nn.Conv1d(int(cfg.hidden_dim), 4, kernel_size=1)

    def _forward_main_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.tcn.base_refines is not None and self.tcn.base_out is not None:
            outputs = [head(refine(hidden)) for refine, head in zip(self.tcn.base_refines, self.tcn.base_out)]
            if self.tcn.out_proj is not None:
                outputs.append(self.tcn.out_proj(hidden))
            return torch.cat(outputs, dim=1)
        return self.tcn.out_proj(hidden)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(batch.shape)}")
        hidden = batch.transpose(1, 2)
        hidden = self.tcn.in_proj(hidden)
        hidden = self.tcn.blocks(hidden)
        logits3 = self._forward_main_from_hidden(hidden).transpose(1, 2)
        logits4 = self.state_head(hidden).transpose(1, 2)
        features = hidden.transpose(1, 2)
        return logits3, logits4, features


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_output_root() -> Path:
    override = os.getenv("AUTORESEARCH_OUTPUT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def _resolve_source_run_dir(cache_root: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    override = os.getenv("AUTORESEARCH_MINDA_SUBASSEMBLY_SOURCE_RUN_DIR", "").strip()
    if override:
        candidates.append(Path(override).expanduser())
    if cache_root is not None:
        manifest_path = cache_root / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for raw in manifest.get("source_run_dirs", []) or []:
                    if str(raw).strip():
                        candidates.append(Path(str(raw)).expanduser())
            except Exception:
                pass
    candidates.extend((DEFAULT_SOURCE_RUN_DIR, LEGACY_SOURCE_RUN_DIR))

    seen: set[str] = set()
    for candidate in candidates:
        dense_root = candidate.resolve() if candidate.exists() else candidate
        key = str(dense_root)
        if key in seen:
            continue
        seen.add(key)
        required = (
            dense_root / "features",
            dense_root / "labels",
            dense_root / "snapshot.json",
            dense_root / "resolved_config.json",
        )
        if all(path.exists() for path in required):
            return dense_root

    raise FileNotFoundError(
        "Unable to resolve a valid subassembly dense_temporal source run. "
        "Run prepare.py first or set AUTORESEARCH_MINDA_SUBASSEMBLY_SOURCE_RUN_DIR."
    )


def _resolve_cache_root() -> Path:
    override = os.getenv("AUTORESEARCH_CACHE_DIR", "").strip()
    base = Path(override).expanduser().resolve() if override else DEFAULT_CACHE_DIR.resolve()
    return base / CACHE_VERSION


def resolve_time_budget_seconds() -> float:
    raw = os.getenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "").strip()
    if not raw:
        return float(DEFAULT_TCN_STAGE_SECONDS + DEFAULT_PROBE_STAGE_SECONDS)
    return max(1.0, float(raw))


def resolve_total_timeout_seconds() -> float:
    raw = os.getenv("AUTORESEARCH_TOTAL_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return float(TOTAL_TIMEOUT_SECONDS)
    return max(resolve_time_budget_seconds(), float(raw))


def resolve_stage_budget_seconds(total_budget_seconds: float) -> Tuple[float, float]:
    tcn_raw = os.getenv("AUTORESEARCH_TCN_STAGE_SECONDS", "").strip()
    probe_raw = os.getenv("AUTORESEARCH_PROBE_STAGE_SECONDS", "").strip()
    tcn_seconds = float(tcn_raw) if tcn_raw else float(DEFAULT_TCN_STAGE_SECONDS)
    probe_seconds = float(probe_raw) if probe_raw else float(DEFAULT_PROBE_STAGE_SECONDS)
    total_raw = os.getenv("AUTORESEARCH_TIME_BUDGET_SECONDS", "").strip()
    if not total_raw:
        return float(tcn_seconds), float(probe_seconds)
    requested = max(1e-6, float(tcn_seconds + probe_seconds))
    scale = float(total_budget_seconds) / float(requested)
    return float(tcn_seconds * scale), float(probe_seconds * scale)


def _ensure_cache(source_run_dir: Path, cache_root: Path) -> Path:
    manifest_path = cache_root / "manifest.json"
    need_build = not manifest_path.exists()
    if not need_build:
        try:
            manifest = load_manifest(cache_root=cache_root)
            sources = {str(Path(item).resolve()) for item in manifest.get("source_run_dirs", [])}
            need_build = (
                str(source_run_dir.resolve()) not in sources
                or str(manifest.get("split_policy") or "").strip() != DEFAULT_SPLIT_POLICY
                or float(manifest.get("val_ratio") or 0.0) != float(VAL_RATIO)
            )
        except Exception:
            need_build = True
    if need_build:
        LOGGER.info(
            "Building cache from %s with split_policy=%s val_ratio=%.2f",
            source_run_dir,
            DEFAULT_SPLIT_POLICY,
            float(VAL_RATIO),
        )
        configure_cache_paths(cache_root)
        build_cache(
            source_run_dirs=[str(source_run_dir)],
            source_globs=[],
            camera_include_regex=None,
            video_include_regex=None,
            path_include_regex=None,
            split_policy=DEFAULT_SPLIT_POLICY,
            val_ratio=float(VAL_RATIO),
            seed=SEED,
            force=True,
        )
    return cache_root


def _video_camera_index(records: Sequence[SegmentRecord]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for record in records:
        current = out.get(record.video_id)
        camera_id = str(record.camera_id)
        if current is None:
            out[record.video_id] = camera_id
            continue
        if current != camera_id:
            raise ValueError(
                f"Video {record.video_id} maps to multiple cameras: {current!r} vs {camera_id!r}"
            )
    return out


def _load_split_plan(cache_root: Path) -> SplitPlan:
    manifest = load_manifest(cache_root=cache_root)
    train_records = tuple(load_split_records("train", cache_root=cache_root))
    val_records = tuple(load_split_records("val", cache_root=cache_root))
    val_eval_records = tuple(load_split_records("val_eval", cache_root=cache_root))
    if not train_records or not val_records or not val_eval_records:
        raise RuntimeError(f"Cache {cache_root} has an empty train/val/val_eval split.")

    train_videos = tuple(sorted({record.video_id for record in train_records}))
    val_videos = tuple(sorted({record.video_id for record in val_records}))
    video_to_camera = _video_camera_index(tuple(train_records) + tuple(val_records))
    camera_total_counts: Dict[str, int] = {}
    for camera_id in video_to_camera.values():
        camera_total_counts[camera_id] = int(camera_total_counts.get(camera_id, 0) + 1)
    camera_val_counts: Dict[str, int] = {}
    for video_id in val_videos:
        camera_id = video_to_camera[video_id]
        camera_val_counts[camera_id] = int(camera_val_counts.get(camera_id, 0) + 1)
    for camera_id in camera_total_counts:
        camera_val_counts.setdefault(camera_id, 0)

    return SplitPlan(
        split_policy=str(manifest.get("split_policy") or DEFAULT_SPLIT_POLICY),
        train_records=train_records,
        val_records=val_records,
        val_eval_records=val_eval_records,
        train_videos=train_videos,
        val_videos=val_videos,
        camera_val_counts={
            str(key): int(value)
            for key, value in (manifest.get("camera_val_counts") or camera_val_counts).items()
        },
        camera_total_counts={
            str(key): int(value)
            for key, value in (manifest.get("camera_total_counts") or camera_total_counts).items()
        },
        target_val_videos=float(manifest.get("split_target_val_videos") or 0.0),
    )


def _resolve_pooler_path(records: Sequence[SegmentRecord]) -> Path:
    for record in records:
        raw = str(record.pooler_checkpoint or "").strip()
        if raw:
            path = Path(raw).expanduser()
            if path.exists():
                return path.resolve()
    fallback = WORKSPACE_ROOT / "encoder_models" / "vjepa2_attention_poolers" / "ssv2-vitl-16x2x3.pt"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("No valid pooler checkpoint found in prepared records.")


def _resolve_encoder_spec(token_dim: int) -> Tuple[str, Path]:
    token_dim_i = int(token_dim)
    if token_dim_i == 1024:
        path = WORKSPACE_ROOT / "encoder_models" / "vitl.pt"
        model = "large"
    elif token_dim_i == 1408:
        path = WORKSPACE_ROOT / "encoder_models" / "vitg-384.pt"
        model = "giant"
    else:
        raise RuntimeError(f"Unsupported token_dim={token_dim_i} for V-JEPA pooler replay")
    if not path.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {path}")
    os.environ["VJEPA_ENCODER_MODEL"] = str(model)
    os.environ["VJEPA_ENCODER_CHECKPOINT"] = str(path)
    os.environ["DENSE_TEMPORAL_ENCODER_MODEL"] = str(model)
    os.environ["DENSE_TEMPORAL_ENCODER_CHECKPOINT"] = str(path)
    return model, path.resolve()


def _build_space_spec(output_root: Path) -> SpaceSpec:
    cache_root = _resolve_cache_root()
    source_run_dir = _resolve_source_run_dir(cache_root)
    cache_root = _ensure_cache(source_run_dir, cache_root)
    split_plan = _load_split_plan(cache_root)
    all_records = tuple(split_plan.train_records) + tuple(split_plan.val_records)
    pooler_path = _resolve_pooler_path(all_records)
    encoder_model, encoder_checkpoint = _resolve_encoder_spec(int(all_records[0].token_dim))
    output_dir = output_root / "subassembly"
    output_dir.mkdir(parents=True, exist_ok=True)
    return SpaceSpec(
        name=SPACE_NAME,
        source_run_dir=source_run_dir,
        cache_root=cache_root,
        output_dir=output_dir,
        split_plan=split_plan,
        pooler_path=pooler_path,
        encoder_model=encoder_model,
        encoder_checkpoint=encoder_checkpoint,
    )


def _torch_from_numpy_safe(
    array: np.ndarray,
    *,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    np_array = np.asarray(array)
    if not np_array.flags.writeable:
        np_array = np_array.copy()
    tensor = torch.from_numpy(np_array).to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _numpy_writable(array: np.ndarray) -> np.ndarray:
    np_array = np.asarray(array)
    if not np_array.flags.writeable:
        np_array = np_array.copy()
    return np_array


def _autocast_context(*, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _clone_tensor_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def _restore_state_dict(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    module.load_state_dict({name: tensor.detach().cpu() for name, tensor in state_dict.items()})


def _update_ema_state(ema_state: Dict[str, torch.Tensor], module: nn.Module, decay: float) -> None:
    alpha = 1.0 - float(decay)
    with torch.no_grad():
        for name, tensor in module.state_dict().items():
            current = tensor.detach().cpu()
            if name not in ema_state:
                ema_state[name] = current.clone()
                continue
            ema_tensor = ema_state[name]
            if current.is_floating_point():
                ema_tensor.mul_(float(decay)).add_(current, alpha=alpha)
            else:
                ema_tensor.copy_(current)


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _count_trainable_params(module: nn.Module) -> int:
    return int(sum(int(param.numel()) for param in module.parameters() if param.requires_grad))


@lru_cache(maxsize=512)
def _load_feature_arrays_cached(feature_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(Path(feature_path)) as data:
        timestamps_ms = np.asarray(data["timestamps_ms"], dtype=np.int64)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    return timestamps_ms, embeddings


def _build_state_targets(
    *,
    length: int,
    mapped_cycles_idx: Sequence[Tuple[int, int]],
    shoulder_radius: int,
    shoulder_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y_state = np.full((int(length),), int(STATE_IDLE), dtype=np.int64)
    weights = np.ones((int(length),), dtype=np.float32)
    radius = int(max(0, shoulder_radius))
    shoulder_w = float(max(0.0, min(1.0, shoulder_weight)))
    for start_idx, end_idx in mapped_cycles_idx:
        s_i = int(start_idx)
        e_i = int(end_idx)
        if s_i < 0 or e_i < 0 or s_i >= int(length) or e_i >= int(length) or e_i < s_i:
            continue
        y_state[s_i : e_i + 1] = int(STATE_WORK)
        y_state[s_i] = int(STATE_START)
        y_state[e_i] = int(STATE_END)
        if radius > 0:
            for idx in range(max(0, s_i - radius), min(int(length), s_i + radius + 1)):
                if idx != s_i:
                    weights[idx] = min(weights[idx], shoulder_w)
            for idx in range(max(0, e_i - radius), min(int(length), e_i + radius + 1)):
                if idx != e_i:
                    weights[idx] = min(weights[idx], shoulder_w)
    return y_state, weights


def _build_supervised_segment(
    record: SegmentRecord,
    *,
    use_eval_span: bool,
    cfg: Stage0Config,
) -> SupervisedSegment:
    if use_eval_span and record.eval_start_idx is not None and record.eval_end_idx is not None:
        global_start_idx = int(record.eval_start_idx)
        global_end_idx = int(record.eval_end_idx)
    else:
        global_start_idx = int(record.supervised_start_idx)
        global_end_idx = int(record.supervised_end_idx)
    if global_end_idx < global_start_idx:
        raise ValueError(f"Invalid slice for {record.segment_id}: {global_start_idx}>{global_end_idx}")

    all_timestamps_ms, all_embeddings = _load_feature_arrays_cached(str(Path(record.feature_path).expanduser().resolve()))
    timestamps_ms = all_timestamps_ms[global_start_idx : global_end_idx + 1].astype(np.int64, copy=False)
    pooled_z0 = all_embeddings[global_start_idx : global_end_idx + 1].astype(np.float32, copy=False)
    gt_pairs = tuple(
        EventPair(start_ms=int(start_ms), end_ms=int(end_ms))
        for start_ms, end_ms in record.event_pairs_ms
        if int(start_ms) >= int(timestamps_ms[0]) and int(end_ms) <= int(timestamps_ms[-1])
    )
    mapped_cycles, _ = map_cycles_to_indices(
        [CycleInterval(start_ms=item.start_ms, end_ms=item.end_ms) for item in gt_pairs],
        timestamps_ms,
        boundary_index_mode=str(cfg.boundary_index_mode),
    )
    y_start, y_end, y_cycle, mask_start_end = build_boundary_targets(
        int(timestamps_ms.shape[0]),
        mapped_cycles,
        ignore_radius=int(cfg.ignore_radius),
        smooth_sigma=float(cfg.smooth_sigma),
    )
    mapped_pairs_idx = tuple((int(item.start_idx), int(item.end_idx)) for item in mapped_cycles)
    y_state, state_weight = _build_state_targets(
        length=int(timestamps_ms.shape[0]),
        mapped_cycles_idx=mapped_pairs_idx,
        shoulder_radius=int(STATE_SHOULDER_RADIUS),
        shoulder_weight=float(STATE_SHOULDER_WEIGHT),
    )
    return SupervisedSegment(
        record=record,
        global_start_idx=int(global_start_idx),
        global_end_idx=int(global_end_idx),
        timestamps_ms=timestamps_ms,
        y_start=y_start.astype(np.float32, copy=False),
        y_end=y_end.astype(np.float32, copy=False),
        y_cycle=y_cycle.astype(np.float32, copy=False),
        mask_start_end=mask_start_end.astype(np.float32, copy=False),
        pooled_z0=pooled_z0,
        y_state=y_state.astype(np.int64, copy=False),
        state_weight=state_weight.astype(np.float32, copy=False),
        mapped_cycles_idx=mapped_pairs_idx,
        gt_pairs=gt_pairs,
    )


def _build_train_sampling_weights(segments: Sequence[SupervisedSegment]) -> np.ndarray:
    if not segments:
        return np.zeros((0,), dtype=np.float64)
    return np.asarray([math.sqrt(max(1, len(segment.gt_pairs))) for segment in segments], dtype=np.float64)


def _build_state_class_weights(segments: Sequence[SupervisedSegment]) -> np.ndarray:
    counts = np.zeros((4,), dtype=np.float64)
    for segment in segments:
        counts += np.bincount(segment.y_state.astype(np.int64, copy=False), minlength=4).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / np.sqrt(counts)
    inv = inv / float(np.mean(inv))
    return inv.astype(np.float32, copy=False)


def _choose_train_segments(
    segments: Sequence[SupervisedSegment],
    *,
    sampling_weights: np.ndarray,
    batch_size: int,
    rng: np.random.RandomState,
) -> List[SupervisedSegment]:
    segment_list = list(segments)
    if not segment_list:
        return []
    probs = np.asarray(sampling_weights, dtype=np.float64)
    if probs.shape != (len(segment_list),) or float(probs.sum()) <= 0.0:
        probs = np.ones((len(segment_list),), dtype=np.float64)
    probs = probs / float(probs.sum())
    batch_size_i = int(batch_size)
    if batch_size_i <= len(segment_list):
        indices = rng.choice(len(segment_list), size=batch_size_i, replace=False, p=probs)
    else:
        unique = rng.choice(len(segment_list), size=len(segment_list), replace=False, p=probs)
        extra = rng.choice(len(segment_list), size=batch_size_i - len(segment_list), replace=True, p=probs)
        indices = np.concatenate((unique, extra), axis=0)
    return [segment_list[int(index)] for index in indices]


def _sample_chunk_bounds(length: int, chunk_len: int, rng: np.random.RandomState) -> Tuple[int, int]:
    if length <= int(chunk_len):
        return 0, int(length) - 1
    start = int(rng.randint(0, int(length) - int(chunk_len) + 1))
    end = start + int(chunk_len) - 1
    return int(start), int(end)


def _fill_halo_row(
    row_np: np.ndarray,
    *,
    segment: SupervisedSegment,
    center_lo: int,
    center_hi: int,
    halo: int,
) -> None:
    _, all_embeddings = _load_feature_arrays_cached(str(Path(segment.record.feature_path).expanduser().resolve()))
    total_windows = int(all_embeddings.shape[0])
    global_center_lo = int(segment.global_start_idx + center_lo)
    global_center_hi = int(segment.global_start_idx + center_hi)
    center_len = int(center_hi - center_lo + 1)

    row_np[int(halo) : int(halo + center_len)] = all_embeddings[global_center_lo : global_center_hi + 1]

    left_src_lo = max(0, int(global_center_lo - halo))
    left_src_hi = int(global_center_lo)
    left_len = max(0, int(left_src_hi - left_src_lo))
    if left_len > 0:
        row_np[int(halo - left_len) : int(halo)] = all_embeddings[left_src_lo:left_src_hi]

    right_src_lo = int(global_center_hi + 1)
    right_src_hi = min(total_windows, int(global_center_hi + 1 + halo))
    right_len = max(0, int(right_src_hi - right_src_lo))
    if right_len > 0:
        row_np[int(halo + center_len) : int(halo + center_len + right_len)] = all_embeddings[right_src_lo:right_src_hi]


def _collate_train_batch(
    segments: Sequence[SupervisedSegment],
    *,
    cfg: Stage0Config,
    sampling_weights: np.ndarray,
    rng: np.random.RandomState,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    chosen: List[Tuple[SupervisedSegment, int, int]] = []
    max_center = 0
    for segment in _choose_train_segments(
        segments,
        sampling_weights=sampling_weights,
        batch_size=int(cfg.batch_size),
        rng=rng,
    ):
        lo, hi = _sample_chunk_bounds(segment.length, int(cfg.chunk_len), rng)
        chosen.append((segment, lo, hi))
        max_center = max(max_center, int(hi - lo + 1))

    total_len = int(HALO_CONTEXT_WINDOWS + max_center + HALO_CONTEXT_WINDOWS)
    feature_batch = torch.zeros((len(chosen), total_len, segments[0].record.embedding_dim), dtype=torch.float32)
    y_state = torch.zeros((len(chosen), total_len), dtype=torch.long)
    state_weight = torch.zeros((len(chosen), total_len), dtype=torch.float32)
    y_cycle = torch.zeros((len(chosen), total_len), dtype=torch.float32)
    y_start = torch.zeros((len(chosen), total_len), dtype=torch.float32)
    y_end = torch.zeros((len(chosen), total_len), dtype=torch.float32)
    mask_start_end = torch.zeros((len(chosen), total_len), dtype=torch.float32)
    valid_mask = torch.zeros((len(chosen), total_len), dtype=torch.float32)

    for row, (segment, lo, hi) in enumerate(chosen):
        center_len = int(hi - lo + 1)
        row_np = np.zeros((total_len, segment.record.embedding_dim), dtype=np.float32)
        _fill_halo_row(
            row_np,
            segment=segment,
            center_lo=lo,
            center_hi=hi,
            halo=int(HALO_CONTEXT_WINDOWS),
        )
        feature_batch[row] = torch.from_numpy(row_np)
        start = int(HALO_CONTEXT_WINDOWS)
        stop = int(HALO_CONTEXT_WINDOWS + center_len)
        y_state[row, start:stop] = torch.from_numpy(segment.y_state[lo : hi + 1])
        state_weight[row, start:stop] = torch.from_numpy(segment.state_weight[lo : hi + 1])
        y_cycle[row, start:stop] = torch.from_numpy(segment.y_cycle[lo : hi + 1])
        y_start[row, start:stop] = torch.from_numpy(segment.y_start[lo : hi + 1])
        y_end[row, start:stop] = torch.from_numpy(segment.y_end[lo : hi + 1])
        mask_start_end[row, start:stop] = torch.from_numpy(segment.mask_start_end[lo : hi + 1])
        valid_mask[row, start:stop] = 1.0

    targets = {
        "y_state": y_state,
        "state_weight": state_weight,
        "y_start": y_start,
        "y_end": y_end,
        "y_cycle": y_cycle,
        "mask_start_end": mask_start_end,
        "valid_mask": valid_mask,
    }
    return feature_batch, targets


def _get_lr_multiplier(*, progress: float, cfg: Stage0Config) -> float:
    warmup = float(cfg.warmup_ratio)
    if progress <= 0.0:
        return 0.0 if warmup > 0.0 else 1.0
    if warmup > 0.0 and progress < warmup:
        return progress / warmup
    tail = float(cfg.final_lr_frac)
    decay_progress = (progress - warmup) / max(1e-6, 1.0 - warmup)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    return 1.0 - (1.0 - tail) * decay_progress


def _compute_transition_consistency_loss(logits: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    cycle_probs = torch.sigmoid(logits[:, :, 2])
    start_probs = torch.sigmoid(logits[:, :, 0])
    end_probs = torch.sigmoid(logits[:, :, 1])

    prev_cycle = F.pad(cycle_probs[:, :-1], (1, 0))
    next_cycle = F.pad(cycle_probs[:, 1:], (0, 1))
    start_from_cycle = (cycle_probs - prev_cycle).clamp_min(0.0)
    end_from_cycle = (cycle_probs - next_cycle).clamp_min(0.0)

    transition_mask = targets["mask_start_end"].to(logits.device) * targets["valid_mask"].to(logits.device)
    start_loss = masked_mean((start_probs - start_from_cycle).square(), transition_mask)
    end_loss = masked_mean((end_probs - end_from_cycle).square(), transition_mask)
    return 0.5 * (start_loss + end_loss)


def _compute_hybrid_stage0_loss(
    logits3: torch.Tensor,
    features: torch.Tensor,
    logits4: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    *,
    cfg: Stage0Config,
    class_weights: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits4 = logits4.float()
    device = logits4.device
    y_state = targets["y_state"].to(device=device)
    state_weight = targets["state_weight"].to(device=device) * targets["valid_mask"].to(device=device)
    ce = F.cross_entropy(
        logits4.transpose(1, 2),
        y_state,
        weight=class_weights.to(device=device),
        reduction="none",
    )
    state_loss = masked_mean(ce, state_weight)
    base_total, base_stats = _compute_stage0_loss(logits3, features, targets, cfg=cfg)
    total = base_total + (float(cfg.aux_loss_weight) * state_loss)
    stats = {
        "loss_total": float(total.detach().item()),
        "loss_boundary": float(base_total.detach().item()),
        "loss_start": float(base_stats["loss_start"]),
        "loss_end": float(base_stats["loss_end"]),
        "loss_cycle": float(base_stats["loss_cycle"]),
        "loss_transition": float(base_stats["loss_transition"]),
        "loss_state": float(state_loss.detach().item()),
    }
    return total, stats


def _compute_stage0_loss(
    logits: torch.Tensor,
    features: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    *,
    cfg: Stage0Config,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    del features
    start_logits = logits[:, :, 0]
    end_logits = logits[:, :, 1]
    cycle_logits = logits[:, :, 2]

    pos_weight_start_end = torch.tensor(float(cfg.pos_weight_start_end), device=logits.device)
    pos_weight_cycle = torch.tensor(float(cfg.pos_weight_cycle), device=logits.device)

    start_loss_raw = focal_bce_with_logits(
        start_logits,
        targets["y_start"].to(logits.device),
        pos_weight=pos_weight_start_end,
        gamma=float(cfg.focal_gamma),
    )
    end_loss_raw = focal_bce_with_logits(
        end_logits,
        targets["y_end"].to(logits.device),
        pos_weight=pos_weight_start_end,
        gamma=float(cfg.focal_gamma),
    )
    cycle_loss_raw = focal_bce_with_logits(
        cycle_logits,
        targets["y_cycle"].to(logits.device),
        pos_weight=pos_weight_cycle,
        gamma=float(cfg.focal_gamma),
    )

    mask_start_end = targets["mask_start_end"].to(logits.device) * targets["valid_mask"].to(logits.device)
    valid_mask = targets["valid_mask"].to(logits.device)
    start_loss = masked_mean(start_loss_raw, mask_start_end)
    end_loss = masked_mean(end_loss_raw, mask_start_end)
    cycle_loss = masked_mean(cycle_loss_raw, valid_mask)
    transition_loss = _compute_transition_consistency_loss(logits, targets)

    total = (
        start_loss
        + end_loss
        + (float(cfg.cycle_loss_weight) * cycle_loss)
        + (float(cfg.transition_consistency_weight) * transition_loss)
    )
    stats = {
        "loss_total": float(total.detach().item()),
        "loss_start": float(start_loss.detach().item()),
        "loss_end": float(end_loss.detach().item()),
        "loss_cycle": float(cycle_loss.detach().item()),
        "loss_aux": 0.0,
        "loss_transition": float(transition_loss.detach().item()),
    }
    return total, stats


def _predict_halo_segment_logits(
    model: Stage0BoundaryModel,
    segment: SupervisedSegment,
    *,
    device: torch.device,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    with torch.inference_mode():
        for center_lo in range(0, int(segment.length), int(HALO_CENTER_CHUNK)):
            center_hi = min(int(segment.length) - 1, int(center_lo + HALO_CENTER_CHUNK - 1))
            center_len = int(center_hi - center_lo + 1)
            row_np = np.zeros((int(HALO_CONTEXT_WINDOWS + center_len + HALO_CONTEXT_WINDOWS), segment.record.embedding_dim), dtype=np.float32)
            _fill_halo_row(
                row_np,
                segment=segment,
                center_lo=center_lo,
                center_hi=center_hi,
                halo=int(HALO_CONTEXT_WINDOWS),
            )
            batch = torch.from_numpy(row_np).unsqueeze(0).to(device=device)
            logits3, _logits4, _features = model(batch)
            logits3 = logits3.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            chunks.append(logits3[int(HALO_CONTEXT_WINDOWS) : int(HALO_CONTEXT_WINDOWS + center_len)])
    if not chunks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _predict_fullseq_logits(
    model: Stage0BoundaryModel,
    segment: SupervisedSegment,
    *,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        batch = torch.from_numpy(_numpy_writable(segment.pooled_z0)).unsqueeze(0).to(device=device)
        logits3, _logits4, _features = model(batch)
        return logits3.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)


def _evaluate_from_main_logits(
    logits_by_segment: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    records: Sequence[SegmentRecord],
) -> Dict[str, float]:
    predictions: Dict[str, Sequence[EventPair]] = {}
    for record in records:
        timestamps_ms, logits3 = logits_by_segment[record.segment_id]
        probs3 = 1.0 / (1.0 + np.exp(-np.asarray(logits3[:, :3], dtype=np.float32)))
        predictions[record.segment_id] = decode_event_pairs(
            probs3,
            timestamps_ms,
            heads=("start", "end", "cycle"),
        )
    return evaluate_predictions(predictions, records=list(records))


def _evaluate_stage0_halo_model(
    model: Stage0BoundaryModel,
    segments: Sequence[SupervisedSegment],
    *,
    device: torch.device,
) -> Dict[str, float]:
    logits_by_segment: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for segment in segments:
        logits_by_segment[segment.record.segment_id] = (
            segment.timestamps_ms,
            _predict_halo_segment_logits(model, segment, device=device),
        )
    return _evaluate_from_main_logits(logits_by_segment, records=[segment.record for segment in segments])


def _evaluate_stage0_fullseq_model(
    model: Stage0BoundaryModel,
    segments: Sequence[SupervisedSegment],
    *,
    device: torch.device,
) -> Dict[str, float]:
    logits_by_segment: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for segment in segments:
        logits_by_segment[segment.record.segment_id] = (
            segment.timestamps_ms,
            _predict_fullseq_logits(model, segment, device=device),
        )
    return _evaluate_from_main_logits(logits_by_segment, records=[segment.record for segment in segments])


def _evaluate_stage0_threshold_proxy(
    model: Stage0BoundaryModel,
    segments: Sequence[SupervisedSegment],
    *,
    device: torch.device,
    prob_threshold: float = VAL_PROXY_THRESHOLD_PROB,
    tolerance_windows: int = VAL_PROXY_TOLERANCE_WINDOWS,
) -> Dict[str, float]:
    start_pred_total = 0
    start_pred_valid = 0
    start_gt_total = 0
    start_gt_covered = 0
    start_duplicate_excess = 0
    end_pred_total = 0
    end_pred_valid = 0
    end_gt_total = 0
    end_gt_covered = 0
    end_duplicate_excess = 0

    threshold = float(prob_threshold)
    tolerance = int(max(0, tolerance_windows))
    for segment in segments:
        logits3 = _predict_halo_segment_logits(model, segment, device=device)
        probs3 = 1.0 / (1.0 + np.exp(-np.asarray(logits3[:, :3], dtype=np.float32)))
        start_mask = probs3[:, 0] > threshold
        end_mask = probs3[:, 1] > threshold

        start_tol_mask, end_tol_mask = _boundary_neighborhood_masks(
            segment.mapped_cycles_idx,
            total_len=int(segment.length),
            radius=int(tolerance),
        )
        start_mask_valid = start_tol_mask > 0.5
        end_mask_valid = end_tol_mask > 0.5
        start_indices = [int(item[0]) for item in segment.mapped_cycles_idx]
        end_indices = [int(item[1]) for item in segment.mapped_cycles_idx]

        start_pred_total += int(start_mask.sum())
        start_pred_valid += int(np.logical_and(start_mask, start_mask_valid).sum())
        start_gt_total += int(len(start_indices))
        covered_start = 0
        for idx in start_indices:
            lo = max(0, int(idx) - tolerance)
            hi = min(int(segment.length), int(idx) + tolerance + 1)
            if bool(start_mask[lo:hi].any()):
                covered_start += 1
        start_gt_covered += int(covered_start)
        start_duplicate_excess += max(0, int(np.logical_and(start_mask, start_mask_valid).sum()) - int(covered_start))

        end_pred_total += int(end_mask.sum())
        end_pred_valid += int(np.logical_and(end_mask, end_mask_valid).sum())
        end_gt_total += int(len(end_indices))
        covered_end = 0
        for idx in end_indices:
            lo = max(0, int(idx) - tolerance)
            hi = min(int(segment.length), int(idx) + tolerance + 1)
            if bool(end_mask[lo:hi].any()):
                covered_end += 1
        end_gt_covered += int(covered_end)
        end_duplicate_excess += max(0, int(np.logical_and(end_mask, end_mask_valid).sum()) - int(covered_end))

    def _f1(*, good_pred: int, total_pred: int, covered_gt: int, total_gt: int) -> Tuple[float, float, float]:
        precision = float(good_pred) / max(1.0, float(total_pred))
        recall = float(covered_gt) / max(1.0, float(total_gt))
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
        return precision, recall, f1

    start_precision, start_recall, start_f1 = _f1(
        good_pred=int(start_pred_valid),
        total_pred=int(start_pred_total),
        covered_gt=int(start_gt_covered),
        total_gt=int(start_gt_total),
    )
    end_precision, end_recall, end_f1 = _f1(
        good_pred=int(end_pred_valid),
        total_pred=int(end_pred_total),
        covered_gt=int(end_gt_covered),
        total_gt=int(end_gt_total),
    )
    return {
        "val_proxy_threshold_prob": float(threshold),
        "val_proxy_tolerance_windows": float(tolerance),
        "val_proxy_start_precision": float(start_precision),
        "val_proxy_start_recall": float(start_recall),
        "val_proxy_start_f1": float(start_f1),
        "val_proxy_start_pred_total": float(start_pred_total),
        "val_proxy_start_pred_valid": float(start_pred_valid),
        "val_proxy_start_gt_total": float(start_gt_total),
        "val_proxy_start_gt_covered": float(start_gt_covered),
        "val_proxy_start_false_count": float(max(0, start_pred_total - start_pred_valid)),
        "val_proxy_start_duplicate_excess": float(start_duplicate_excess),
        "val_proxy_end_precision": float(end_precision),
        "val_proxy_end_recall": float(end_recall),
        "val_proxy_end_f1": float(end_f1),
        "val_proxy_end_pred_total": float(end_pred_total),
        "val_proxy_end_pred_valid": float(end_pred_valid),
        "val_proxy_end_gt_total": float(end_gt_total),
        "val_proxy_end_gt_covered": float(end_gt_covered),
        "val_proxy_end_false_count": float(max(0, end_pred_total - end_pred_valid)),
        "val_proxy_end_duplicate_excess": float(end_duplicate_excess),
        "val_proxy_macro_f1": float(0.5 * (float(start_f1) + float(end_f1))),
        "val_proxy_total_false_count": float(
            max(0, start_pred_total - start_pred_valid) + max(0, end_pred_total - end_pred_valid)
        ),
    }


def _stage0_selection_score(*, halo_metrics: Dict[str, float], proxy_metrics: Dict[str, float]) -> float:
    return 0.5 * (float(halo_metrics["val_pair_f1"]) + float(proxy_metrics["val_proxy_macro_f1"]))


def _stage0_selection_metrics(
    *,
    halo_metrics: Dict[str, float],
    fullseq_metrics: Dict[str, float],
    proxy_metrics: Dict[str, float],
) -> Dict[str, float]:
    return {
        "val_selection_score": float(_stage0_selection_score(halo_metrics=halo_metrics, proxy_metrics=proxy_metrics)),
        "val_halo16_pair_f1": float(halo_metrics["val_pair_f1"]),
        "val_halo16_pair_precision": float(halo_metrics["val_pair_precision"]),
        "val_halo16_pair_recall": float(halo_metrics["val_pair_recall"]),
        "val_halo16_count_mae": float(halo_metrics["val_count_mae"]),
        "val_halo16_start_mae_ms": float(halo_metrics["val_start_mae_ms"]),
        "val_halo16_end_mae_ms": float(halo_metrics["val_end_mae_ms"]),
        "val_fullseq_pair_f1": float(fullseq_metrics["val_pair_f1"]),
        "val_fullseq_pair_precision": float(fullseq_metrics["val_pair_precision"]),
        "val_fullseq_pair_recall": float(fullseq_metrics["val_pair_recall"]),
        "val_fullseq_count_mae": float(fullseq_metrics["val_count_mae"]),
        "val_fullseq_start_mae_ms": float(fullseq_metrics["val_start_mae_ms"]),
        "val_fullseq_end_mae_ms": float(fullseq_metrics["val_end_mae_ms"]),
        "val_proxy_macro_f1": float(proxy_metrics["val_proxy_macro_f1"]),
        "val_proxy_total_false_count": float(proxy_metrics["val_proxy_total_false_count"]),
    }


def _stage0_component_tuple(payload: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    halo_metrics = payload["halo16_eval"]["metrics"]
    fullseq_metrics = payload["fullseq_eval"]["metrics"]
    proxy_metrics = payload["threshold_proxy_eval"]
    return (
        float(_stage0_selection_score(halo_metrics=halo_metrics, proxy_metrics=proxy_metrics)),
        float(halo_metrics["val_pair_f1"]),
        float(proxy_metrics["val_proxy_macro_f1"]),
        float(proxy_metrics["val_proxy_total_false_count"]),
        float(fullseq_metrics["val_pair_f1"]),
    )


def _is_better_stage0_eval(candidate: Dict[str, Any], incumbent: Optional[Dict[str, Any]]) -> bool:
    if incumbent is None:
        return True
    eps = 1e-9
    candidate_avg, candidate_halo, candidate_proxy, candidate_false, candidate_fullseq = _stage0_component_tuple(candidate)
    incumbent_avg, incumbent_halo, incumbent_proxy, incumbent_false, incumbent_fullseq = _stage0_component_tuple(incumbent)

    material_drop = (
        candidate_halo < (incumbent_halo - float(STAGE0_MATERIAL_DEGRADE_EPS))
        or candidate_proxy < (incumbent_proxy - float(STAGE0_MATERIAL_DEGRADE_EPS))
    )
    if candidate_avg > incumbent_avg + eps:
        if material_drop:
            return False
        return True
    if candidate_avg < incumbent_avg - eps:
        return False
    if material_drop:
        return False
    if candidate_false < incumbent_false - eps:
        return True
    if candidate_false > incumbent_false + eps:
        return False
    if candidate_fullseq > incumbent_fullseq + eps:
        return True
    if candidate_fullseq < incumbent_fullseq - eps:
        return False
    if candidate_halo > incumbent_halo + eps:
        return True
    if candidate_halo < incumbent_halo - eps:
        return False
    return candidate_proxy > incumbent_proxy + eps


def _checkpoint_decode_heads(payload: Dict[str, Any]) -> Tuple[List[str], int]:
    heads = [str(head).strip().lower() for head in payload.get("heads", []) if str(head).strip()]
    if "boundary" in heads and "start" not in heads and "end" not in heads:
        return ["boundary", "cycle"], 2
    return ["start", "end", "cycle"], 3


def _eval_gt_pairs_for_timestamps(
    record: SegmentRecord,
    *,
    timestamps_ms: np.ndarray,
) -> Tuple[EventPair, ...]:
    if int(timestamps_ms.shape[0]) <= 0:
        return ()
    return tuple(
        EventPair(start_ms=int(start_ms), end_ms=int(end_ms))
        for start_ms, end_ms in record.event_pairs_ms
        if int(start_ms) >= int(timestamps_ms[0]) and int(end_ms) <= int(timestamps_ms[-1])
    )


def _map_eval_cycles_idx(
    record: SegmentRecord,
    *,
    timestamps_ms: np.ndarray,
    boundary_index_mode: str,
) -> Tuple[Tuple[int, int], ...]:
    gt_pairs = _eval_gt_pairs_for_timestamps(record, timestamps_ms=timestamps_ms)
    mapped_cycles, _ = map_cycles_to_indices(
        [CycleInterval(start_ms=item.start_ms, end_ms=item.end_ms) for item in gt_pairs],
        timestamps_ms,
        boundary_index_mode=str(boundary_index_mode),
    )
    return tuple((int(item.start_idx), int(item.end_idx)) for item in mapped_cycles)


def _boundary_neighborhood_masks(
    mapped_cycles_idx: Sequence[Tuple[int, int]],
    *,
    total_len: int,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray]:
    chunk_len = int(max(0, total_len))
    start_mask = np.zeros((chunk_len,), dtype=np.float32)
    end_mask = np.zeros((chunk_len,), dtype=np.float32)
    tol = int(max(0, radius))
    for cycle_start, cycle_end in mapped_cycles_idx:
        s_i = int(cycle_start)
        e_i = int(cycle_end)
        for idx in range(max(0, s_i - tol), min(chunk_len, s_i + tol + 1)):
            start_mask[int(idx)] = 1.0
        for idx in range(max(0, e_i - tol), min(chunk_len, e_i + tol + 1)):
            end_mask[int(idx)] = 1.0
    return start_mask, end_mask


def _proxy_start_end_probs(probs: np.ndarray, *, heads: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    head_names = [str(head).strip().lower() for head in heads]
    head_to_idx = {name: idx for idx, name in enumerate(head_names)}
    if "start" in head_to_idx and "end" in head_to_idx:
        return probs[:, int(head_to_idx["start"])], probs[:, int(head_to_idx["end"])]
    if "boundary" in head_to_idx:
        boundary = probs[:, int(head_to_idx["boundary"])]
        return boundary, boundary
    raise RuntimeError(f"Unable to resolve start/end heads from checkpoint heads={head_names!r}")


def _evaluate_threshold_proxy_segments(
    segments: Sequence[ValidationEvalSegment],
    *,
    heads: Sequence[str],
    prob_threshold: float = VAL_PROXY_THRESHOLD_PROB,
    tolerance_windows: int = VAL_PROXY_TOLERANCE_WINDOWS,
) -> Dict[str, float]:
    start_pred_total = 0
    start_pred_valid = 0
    start_gt_total = 0
    start_gt_covered = 0
    start_duplicate_excess = 0

    end_pred_total = 0
    end_pred_valid = 0
    end_gt_total = 0
    end_gt_covered = 0
    end_duplicate_excess = 0

    threshold = float(prob_threshold)
    tolerance = int(max(0, tolerance_windows))
    base_heads = int(len(heads))
    for segment in segments:
        if int(segment.logits.shape[0]) != int(segment.timestamps_ms.shape[0]):
            raise RuntimeError(
                f"Validation logits length mismatch for {segment.record.segment_id}: "
                f"{segment.logits.shape[0]} vs {segment.timestamps_ms.shape[0]}"
            )
        probs = 1.0 / (1.0 + np.exp(-segment.logits[:, :base_heads]))
        start_probs, end_probs = _proxy_start_end_probs(probs, heads=heads)
        start_mask = start_probs > threshold
        end_mask = end_probs > threshold
        start_indices = [int(item[0]) for item in segment.mapped_cycles_idx]
        end_indices = [int(item[1]) for item in segment.mapped_cycles_idx]
        start_tol_mask, end_tol_mask = _boundary_neighborhood_masks(
            segment.mapped_cycles_idx,
            total_len=int(segment.timestamps_ms.shape[0]),
            radius=int(tolerance),
        )
        start_mask_valid = start_tol_mask > 0.5
        end_mask_valid = end_tol_mask > 0.5

        start_pred_total += int(start_mask.sum())
        start_pred_valid += int(np.logical_and(start_mask, start_mask_valid).sum())
        start_gt_total += int(len(start_indices))
        covered_start = 0
        for idx in start_indices:
            lo = max(0, int(idx) - tolerance)
            hi = min(int(segment.timestamps_ms.shape[0]), int(idx) + tolerance + 1)
            if bool(start_mask[lo:hi].any()):
                covered_start += 1
        start_gt_covered += int(covered_start)
        start_duplicate_excess += max(
            0,
            int(np.logical_and(start_mask, start_mask_valid).sum()) - int(covered_start),
        )

        end_pred_total += int(end_mask.sum())
        end_pred_valid += int(np.logical_and(end_mask, end_mask_valid).sum())
        end_gt_total += int(len(end_indices))
        covered_end = 0
        for idx in end_indices:
            lo = max(0, int(idx) - tolerance)
            hi = min(int(segment.timestamps_ms.shape[0]), int(idx) + tolerance + 1)
            if bool(end_mask[lo:hi].any()):
                covered_end += 1
        end_gt_covered += int(covered_end)
        end_duplicate_excess += max(
            0,
            int(np.logical_and(end_mask, end_mask_valid).sum()) - int(covered_end),
        )

    def _f1(*, good_pred: int, total_pred: int, covered_gt: int, total_gt: int) -> Tuple[float, float, float]:
        precision = float(good_pred) / max(1.0, float(total_pred))
        recall = float(covered_gt) / max(1.0, float(total_gt))
        f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
        return precision, recall, f1

    start_precision, start_recall, start_f1 = _f1(
        good_pred=int(start_pred_valid),
        total_pred=int(start_pred_total),
        covered_gt=int(start_gt_covered),
        total_gt=int(start_gt_total),
    )
    end_precision, end_recall, end_f1 = _f1(
        good_pred=int(end_pred_valid),
        total_pred=int(end_pred_total),
        covered_gt=int(end_gt_covered),
        total_gt=int(end_gt_total),
    )
    return {
        "val_proxy_threshold_prob": float(threshold),
        "val_proxy_tolerance_windows": float(tolerance),
        "val_proxy_start_precision": float(start_precision),
        "val_proxy_start_recall": float(start_recall),
        "val_proxy_start_f1": float(start_f1),
        "val_proxy_start_pred_total": float(start_pred_total),
        "val_proxy_start_pred_valid": float(start_pred_valid),
        "val_proxy_start_gt_total": float(start_gt_total),
        "val_proxy_start_gt_covered": float(start_gt_covered),
        "val_proxy_start_false_count": float(max(0, start_pred_total - start_pred_valid)),
        "val_proxy_start_duplicate_excess": float(start_duplicate_excess),
        "val_proxy_end_precision": float(end_precision),
        "val_proxy_end_recall": float(end_recall),
        "val_proxy_end_f1": float(end_f1),
        "val_proxy_end_pred_total": float(end_pred_total),
        "val_proxy_end_pred_valid": float(end_pred_valid),
        "val_proxy_end_gt_total": float(end_gt_total),
        "val_proxy_end_gt_covered": float(end_gt_covered),
        "val_proxy_end_false_count": float(max(0, end_pred_total - end_pred_valid)),
        "val_proxy_end_duplicate_excess": float(end_duplicate_excess),
        "val_proxy_macro_f1": float(0.5 * (float(start_f1) + float(end_f1))),
        "val_proxy_total_false_count": float(
            max(0, start_pred_total - start_pred_valid) + max(0, end_pred_total - end_pred_valid)
        ),
    }


def _merge_validation_metrics(
    *,
    legacy_metrics: Dict[str, float],
    proxy_metrics: Dict[str, float],
) -> Dict[str, float]:
    merged = dict(legacy_metrics)
    merged["val_legacy_pair_f1"] = float(legacy_metrics["val_pair_f1"])
    merged["val_legacy_pair_precision"] = float(legacy_metrics["val_pair_precision"])
    merged["val_legacy_pair_recall"] = float(legacy_metrics["val_pair_recall"])
    merged.update(proxy_metrics)
    merged["val_pair_f1"] = float(proxy_metrics["val_proxy_macro_f1"])
    merged["val_pair_precision"] = float(
        0.5 * (float(proxy_metrics["val_proxy_start_precision"]) + float(proxy_metrics["val_proxy_end_precision"]))
    )
    merged["val_pair_recall"] = float(
        0.5 * (float(proxy_metrics["val_proxy_start_recall"]) + float(proxy_metrics["val_proxy_end_recall"]))
    )
    return merged


def _evaluate_validation_segments(
    segments: Sequence[ValidationEvalSegment],
    *,
    decode_heads: Sequence[str],
) -> Dict[str, float]:
    predictions: Dict[str, Sequence[EventPair]] = {}
    for segment in segments:
        probs = 1.0 / (1.0 + np.exp(-segment.logits[:, : len(decode_heads)]))
        predictions[segment.record.segment_id] = decode_event_pairs(
            probs,
            segment.timestamps_ms,
            heads=decode_heads,
        )
    legacy_metrics = evaluate_predictions(predictions, records=[segment.record for segment in segments])
    proxy_metrics = _evaluate_threshold_proxy_segments(segments, heads=decode_heads)
    return _merge_validation_metrics(legacy_metrics=legacy_metrics, proxy_metrics=proxy_metrics)


def _predict_halo_logits_from_pooled(
    model: nn.Module,
    pooled: np.ndarray,
    *,
    device: torch.device,
    halo_context: int = HALO_CONTEXT_WINDOWS,
    center_chunk: int = HALO_CENTER_CHUNK,
) -> np.ndarray:
    seq_len = int(pooled.shape[0])
    if seq_len <= 0:
        return np.zeros((0, 0), dtype=np.float32)
    feat_dim = int(pooled.shape[1])
    halo = int(max(0, halo_context))
    center = int(max(1, center_chunk))
    outputs: Optional[np.ndarray] = None
    model.eval()
    with torch.inference_mode():
        for center_start in range(0, seq_len, center):
            center_end = min(seq_len, int(center_start + center))
            center_len = int(center_end - center_start)
            block = np.zeros((halo + center_len + halo, feat_dim), dtype=np.float32)

            left_src_lo = max(0, int(center_start - halo))
            left_src_hi = int(center_start)
            left_len = max(0, int(left_src_hi - left_src_lo))
            if left_len > 0:
                block[int(halo - left_len) : int(halo)] = pooled[left_src_lo:left_src_hi]

            block[int(halo) : int(halo + center_len)] = pooled[center_start:center_end]

            right_src_lo = int(center_end)
            right_src_hi = min(seq_len, int(center_end + halo))
            right_len = max(0, int(right_src_hi - right_src_lo))
            if right_len > 0:
                block[int(halo + center_len) : int(halo + center_len + right_len)] = pooled[right_src_lo:right_src_hi]

            batch = torch.from_numpy(block).unsqueeze(0).to(device=device)
            logits = model(batch).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
            if outputs is None:
                outputs = np.zeros((seq_len, int(logits.shape[1])), dtype=np.float32)
            outputs[center_start:center_end] = logits[int(halo) : int(halo + center_len)]
    if outputs is None:
        return np.zeros((seq_len, 0), dtype=np.float32)
    return outputs


def _evaluate_pooled_checkpoint(
    checkpoint_path: Path,
    records: Sequence[SegmentRecord],
    *,
    device: torch.device,
) -> Dict[str, float]:
    if not records:
        raise RuntimeError("Validation records are required for pooled checkpoint evaluation")
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = prod_tcn.load_boundary_checkpoint(
        checkpoint_path,
        input_dim=int(records[0].embedding_dim),
        device=str(device),
    )
    decode_heads, base_heads = _checkpoint_decode_heads(payload)
    train_cfg = payload.get("train_cfg", {}) if isinstance(payload.get("train_cfg"), dict) else {}
    boundary_index_mode = str(train_cfg.get("boundary_index_mode", "ordered_nearest"))
    eval_segments: List[ValidationEvalSegment] = []
    with torch.inference_mode():
        for record in records:
            arrays = load_segment_arrays(record, representation="pooled_z0", use_eval_span=True)
            logits = _predict_halo_logits_from_pooled(
                model,
                arrays["pooled_z0"],
                device=device,
            )
            eval_segments.append(
                ValidationEvalSegment(
                    record=record,
                    timestamps_ms=arrays["timestamps_ms"],
                    logits=logits[:, :base_heads],
                    mapped_cycles_idx=_map_eval_cycles_idx(
                        record,
                        timestamps_ms=arrays["timestamps_ms"],
                        boundary_index_mode=boundary_index_mode,
                    ),
                )
            )
    return _evaluate_validation_segments(eval_segments, decode_heads=decode_heads)


def _encode_tokens_with_probe(probe: nn.Module, tokens: np.ndarray, *, device: torch.device) -> np.ndarray:
    if int(tokens.shape[0]) <= 0:
        return np.zeros((0, int(tokens.shape[-1])), dtype=np.float32)
    outputs: List[np.ndarray] = []
    use_autocast = device.type == "cuda"
    probe.eval()
    with torch.inference_mode():
        for lo in range(0, int(tokens.shape[0]), int(EVAL_TOKEN_CHUNK)):
            hi = min(int(tokens.shape[0]), lo + int(EVAL_TOKEN_CHUNK))
            batch = torch.from_numpy(tokens[lo:hi]).to(device=device)
            autocast_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_autocast else nullcontext()
            with autocast_ctx:
                pooled = probe(batch).squeeze(1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, int(tokens.shape[-1])), dtype=np.float32)


def _compute_timing_mae_ms(metrics: Dict[str, float]) -> float:
    return 0.5 * (float(metrics["val_start_mae_ms"]) + float(metrics["val_end_mae_ms"]))


def _primary_pair_f1(metrics: Dict[str, float]) -> float:
    return float(metrics.get("val_proxy_macro_f1", metrics["val_pair_f1"]))


def _legacy_pair_f1(metrics: Dict[str, float]) -> float:
    return float(metrics.get("val_legacy_pair_f1", metrics["val_pair_f1"]))


def _is_better_count_then_timing(candidate: Dict[str, float], incumbent: Dict[str, float]) -> bool:
    eps = 1e-9
    candidate_count = float(candidate["val_count_mae"])
    incumbent_count = float(incumbent["val_count_mae"])
    if candidate_count < incumbent_count - eps:
        return True
    if candidate_count > incumbent_count + eps:
        return False
    return _compute_timing_mae_ms(candidate) < (_compute_timing_mae_ms(incumbent) - eps)


def _is_better_metrics(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    if incumbent is None:
        return True
    eps = 1e-9
    candidate_f1 = _primary_pair_f1(candidate)
    incumbent_f1 = _primary_pair_f1(incumbent)
    if candidate_f1 > incumbent_f1 + eps:
        return True
    if candidate_f1 < incumbent_f1 - eps:
        return False
    if "val_proxy_macro_f1" in candidate and "val_proxy_macro_f1" in incumbent:
        candidate_false = float(candidate.get("val_proxy_total_false_count", 0.0))
        incumbent_false = float(incumbent.get("val_proxy_total_false_count", 0.0))
        if candidate_false < incumbent_false - eps:
            return True
        if candidate_false > incumbent_false + eps:
            return False
        candidate_legacy = _legacy_pair_f1(candidate)
        incumbent_legacy = _legacy_pair_f1(incumbent)
        if candidate_legacy > incumbent_legacy + eps:
            return True
        if candidate_legacy < incumbent_legacy - eps:
            return False
    return _is_better_count_then_timing(candidate, incumbent)


def _stage0_train_cfg_dict(cfg: Stage0Config, *, steps_per_epoch: int) -> Dict[str, Any]:
    return {
        "seq_model": str(cfg.seq_model),
        "hidden_dim": int(cfg.hidden_dim),
        "kernel_size": int(cfg.kernel_size),
        "dropout": float(cfg.dropout),
        "use_layernorm": bool(cfg.use_layernorm),
        "dilations": tuple(int(item) for item in cfg.dilations),
        "bidirectional": bool(cfg.bidirectional),
        "task_specific_heads": bool(cfg.task_specific_heads),
        "boundary_loss": "focal",
        "gamma": float(cfg.focal_gamma),
        "pos_weight_start_end": float(cfg.pos_weight_start_end),
        "pos_weight_cycle": float(cfg.pos_weight_cycle),
        "use_phase_head": False,
        "phase_loss_weight": 0.0,
        "phase_loss": "mse",
        "phase_huber_delta": 0.25,
        "ranking_loss_weight": 0.0,
        "ranking_margin": float(cfg.neg_margin),
        "ranking_window_radius": 1,
        "cyclece_loss_weight": 0.0,
        "cyclece_tau": 0.7,
        "cyclece_radius": 1,
        "class_loss_weight": 0.0,
        "class_sampling_alpha": 0.35,
        "temporal_structure_mode": "cyclic",
        "ignore_radius": int(cfg.ignore_radius),
        "smooth_sigma": float(cfg.smooth_sigma),
        "combine_start_end": bool(cfg.combine_start_end),
        "boundary_index_mode": str(cfg.boundary_index_mode),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "max_epochs": int(cfg.epochs),
        "seed": int(cfg.seed),
        "device": "cuda",
        "grad_clip_norm": float(cfg.grad_clip_norm),
        "chunk_len": int(cfg.chunk_len),
        "chunks_per_epoch": int(steps_per_epoch),
        "neg_chunk_fraction": 0.0,
        "stage1_epochs": 20,
        "stage1_probe_lr": 1.5e-5,
        "stage1_last_block_epoch": 0,
        "stage1_all_epoch": 999,
        "stage1_chunks_per_stream": 12,
        "stage1_tcn_tune_mode": "frozen",
        "stage1_tcn_last_blocks": 1,
        "stage1_tcn_lr": 5e-5,
        "stage1_stream_sampling_mode": "uniform",
        "stage1_stream_sampling_power": 0.5,
        "stage1_stream_sampling_min_weight": 1.0,
        "stage1_cyclece_weight": 1.0,
        "stage1_smooth_weight": 0.05,
        "stage1_distill_weight": 0.10,
        "stage1_class_weight": 0.0,
    }


def _save_stage0_checkpoint(
    checkpoint_path: Path,
    *,
    model: Stage0BoundaryModel,
    cfg: Stage0Config,
    metrics: Dict[str, float],
    best_epoch: int,
    best_source: str,
    steps_per_epoch: int,
    total_steps: int,
    train_records: Sequence[SegmentRecord],
    val_records: Sequence[SegmentRecord],
    pooler_path: Path,
    cache_root: Path,
    state_class_weights: np.ndarray,
) -> None:
    model_cfg = BoundaryTCNConfig(
        input_dim=int(train_records[0].embedding_dim),
        hidden_dim=int(cfg.hidden_dim),
        out_dim=3,
        kernel_size=int(cfg.kernel_size),
        dropout=float(cfg.dropout),
        use_layernorm=bool(cfg.use_layernorm),
        dilations=tuple(int(item) for item in cfg.dilations),
        bidirectional=bool(cfg.bidirectional),
        task_specific_heads=bool(cfg.task_specific_heads),
        base_heads=3,
    )
    train_cfg = _stage0_train_cfg_dict(cfg, steps_per_epoch=steps_per_epoch)
    payload: Dict[str, Any] = {
        "seq_model": "tcn",
        "model_state": {name: tensor.detach().cpu() for name, tensor in model.tcn.state_dict().items()},
        "aux_state_head_state": {name: tensor.detach().cpu() for name, tensor in model.state_head.state_dict().items()},
        "model_cfg": asdict(model_cfg),
        "train_cfg": train_cfg,
        "heads": ["start", "end", "cycle"],
        "effective_losses": {
            "boundary": True,
            "cycle": True,
            "phase": False,
            "ranking": False,
            "cyclece": False,
            "class": False,
            "transition_consistency": True,
        },
        "multiclass_head": {
            "enabled": False,
            "class_offset": 3,
            "num_classes": 0,
            "class_index_to_label_id": [],
            "class_index_to_label_name": [],
            "label_id_to_label_name": {},
            "action_label_ids": [],
            "loss_weight": 0.0,
            "class_weights": [],
        },
        "representation_mode": "pooled_z0",
        "pooler_tune_mode": "off",
        "pooler_checkpoint": str(pooler_path),
        "pooler_sha": str(train_records[0].pooler_sha),
        "model_family": str(cfg.model_family),
        "metrics": dict(metrics),
        "pos_weight_cycle": float(cfg.pos_weight_cycle),
        "pos_weight_start_end": float(cfg.pos_weight_start_end),
        "best_epoch": int(best_epoch),
        "best_source": str(best_source),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "cache_root": str(cache_root),
        "train_segments": int(len(train_records)),
        "val_segments": int(len(val_records)),
        "replay_contract": {
            "faithful_stage0_recipe": True,
            "intentional_deviations": [
                "wallclock_limited_stage0",
                "camera_stratified_hash_60_40_split",
            ],
        },
        "rd_experiment": {
            "name": "halo16_hybrid_stage0",
            "halo": int(HALO_CONTEXT_WINDOWS),
            "center_chunk_len": int(HALO_CENTER_CHUNK),
            "selection_metric": "guarded_avg_halo16_pair_f1_proxy_macro_f1_then_false_then_fullseq",
            "primary_heads": ["start", "end", "cycle"],
            "aux_state_head": list(STATE_NAMES),
            "state_class_weights": [float(item) for item in state_class_weights.tolist()],
            "state_aux_weight": float(cfg.aux_loss_weight),
            "shoulder_radius": int(STATE_SHOULDER_RADIUS),
            "shoulder_weight": float(STATE_SHOULDER_WEIGHT),
        },
    }
    payload["boundary_arch_version"] = prod_tcn.infer_boundary_arch_version(
        model_cfg=payload["model_cfg"],
        model_state=payload["model_state"],
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def _run_stage0(
    *,
    spec: SpaceSpec,
    output_dir: Path,
    cfg: Stage0Config,
    budget_seconds: float,
    device: torch.device,
) -> Stage0Result:
    train_records = spec.split_plan.train_records
    val_records = spec.split_plan.val_eval_records
    if not train_records or not val_records:
        raise RuntimeError("Stage-0 requires non-empty train and val record sets")

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(cfg.seed))
    rng = np.random.RandomState(int(cfg.seed))

    train_segments = [
        _build_supervised_segment(record, use_eval_span=False, cfg=cfg)
        for record in train_records
    ]
    val_segments = [
        _build_supervised_segment(record, use_eval_span=True, cfg=cfg)
        for record in val_records
    ]
    sampling_weights = _build_train_sampling_weights(train_segments)
    state_class_weights_np = _build_state_class_weights(train_segments)
    state_class_weights_t = torch.from_numpy(state_class_weights_np).to(device=device)
    model = Stage0BoundaryModel(input_dim=int(train_records[0].embedding_dim), cfg=cfg).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for stage-0 replay")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    use_autocast = device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_autocast else torch.float32

    steps_per_epoch = int(max(1, math.ceil(len(train_segments) / float(cfg.batch_size))))
    total_steps = int(max(1, int(cfg.epochs) * steps_per_epoch))
    LOGGER.info(
        "Stage-0 hybrid: train_segments=%d val_segments=%d epochs=%d steps_per_epoch=%d total_steps=%d budget=%.1fs halo=%d center_chunk_len=%d state_aux_weight=%.4f class_weights=%s",
        len(train_segments),
        len(val_segments),
        int(cfg.epochs),
        int(steps_per_epoch),
        int(total_steps),
        float(budget_seconds),
        int(HALO_CONTEXT_WINDOWS),
        int(HALO_CENTER_CHUNK),
        float(cfg.aux_loss_weight),
        [round(float(item), 4) for item in state_class_weights_np.tolist()],
    )

    deadline = time.monotonic() + float(budget_seconds)
    history: List[Dict[str, Any]] = []
    best_eval_payload: Optional[Dict[str, Any]] = None
    best_eval_state: Optional[Dict[str, torch.Tensor]] = None
    best_state_head: Optional[Dict[str, torch.Tensor]] = None
    best_source = "none"
    best_epoch = -1
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    peak_vram_mb = 0.0
    global_step = 0
    start_t = time.time()

    for epoch in range(int(cfg.epochs)):
        model.train()
        epoch_totals = {
            "loss_total": 0.0,
            "loss_boundary": 0.0,
            "loss_start": 0.0,
            "loss_end": 0.0,
            "loss_state": 0.0,
            "loss_cycle": 0.0,
            "loss_transition": 0.0,
        }
        epoch_t0 = time.time()
        completed_steps = 0

        for _ in range(int(steps_per_epoch)):
            if completed_steps > 0 and time.monotonic() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            running_stats = {key: 0.0 for key in epoch_totals}
            progress = float(global_step) / float(max(1, total_steps - 1))
            for _micro_step in range(int(cfg.grad_accum_steps)):
                features_cpu, targets = _collate_train_batch(
                    train_segments,
                    cfg=cfg,
                    sampling_weights=sampling_weights,
                    rng=rng,
                )
                batch = features_cpu.to(device=device, non_blocking=True)
                targets_t = {key: value.to(device=device, non_blocking=True) for key, value in targets.items()}
                autocast_ctx = torch.autocast(device_type="cuda", dtype=autocast_dtype) if use_autocast else nullcontext()
                with autocast_ctx:
                    logits3, logits4, features = model(batch)
                    loss, stats = _compute_hybrid_stage0_loss(
                        logits3,
                        features,
                        logits4,
                        targets_t,
                        cfg=cfg,
                        class_weights=state_class_weights_t,
                    )
                    loss = loss / float(cfg.grad_accum_steps)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch}")
                loss.backward()
                for key, value in stats.items():
                    running_stats[key] += float(value)

            if float(cfg.grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.grad_clip_norm))
            lr_mult = _get_lr_multiplier(progress=progress, cfg=cfg)
            for group in optimizer.param_groups:
                group["lr"] = float(cfg.lr) * float(lr_mult)
            optimizer.step()

            global_step += 1
            completed_steps += 1
            ema_progress = float(global_step) / float(max(1, total_steps))
            if bool(cfg.ema_enabled):
                if ema_state is None and ema_progress >= float(cfg.ema_start_ratio):
                    ema_state = _clone_state_dict(model.tcn)
                elif ema_state is not None:
                    _update_ema_state(ema_state, model.tcn, float(cfg.ema_decay))

            for key, value in running_stats.items():
                epoch_totals[key] += float(value / float(max(1, cfg.grad_accum_steps)))

            if device.type == "cuda":
                peak_vram_mb = max(peak_vram_mb, float(torch.cuda.max_memory_allocated(device)) / 1024.0 / 1024.0)

            if global_step % int(cfg.log_every_steps) == 0:
                LOGGER.info(
                    "stage0 step=%04d/%04d epoch=%03d loss=%.4f boundary=%.4f state=%.4f start=%.4f end=%.4f cycle=%.4f transition=%.4f lr=%.6g",
                    global_step,
                    total_steps,
                    epoch,
                    running_stats["loss_total"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_boundary"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_state"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_start"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_end"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_cycle"] / float(max(1, cfg.grad_accum_steps)),
                    running_stats["loss_transition"] / float(max(1, cfg.grad_accum_steps)),
                    optimizer.param_groups[0]["lr"],
                )

        if completed_steps <= 0:
            break

        epoch_row: Dict[str, Any] = {
            "epoch": int(epoch),
            "steps": int(completed_steps),
            "seconds": float(time.time() - epoch_t0),
        }
        for key, value in epoch_totals.items():
            epoch_row[key] = float(value / float(max(1, completed_steps)))

        should_eval = ((epoch + 1) % int(cfg.val_every_epochs) == 0) or (time.monotonic() >= deadline)
        if should_eval:
            model.eval()
            current_halo = _evaluate_stage0_halo_model(model, val_segments, device=device)
            current_fullseq = _evaluate_stage0_fullseq_model(model, val_segments, device=device)
            current_proxy = _evaluate_stage0_threshold_proxy(model, val_segments, device=device)
            current_payload = {
                "epoch": int(epoch),
                "source": "current",
                "halo16_eval": {"metrics": current_halo, "timing_mae_ms": float(_compute_timing_mae_ms(current_halo))},
                "fullseq_eval": {
                    "metrics": current_fullseq,
                    "timing_mae_ms": float(_compute_timing_mae_ms(current_fullseq)),
                },
                "threshold_proxy_eval": current_proxy,
            }
            epoch_row["val_current"] = current_payload
            improved = _is_better_stage0_eval(current_payload, best_eval_payload)
            if improved:
                best_eval_payload = current_payload
                best_eval_state = _clone_state_dict(model.tcn)
                best_state_head = _clone_state_dict(model.state_head)
                best_epoch = int(epoch)
                best_source = "current"

            LOGGER.info(
                "stage0 val epoch=%03d source=current score=%.6f halo16=%.6f proxy=%.6f false=%d fullseq=%.6f %s",
                epoch,
                _stage0_selection_score(halo_metrics=current_halo, proxy_metrics=current_proxy),
                float(current_halo["val_pair_f1"]),
                float(current_proxy["val_proxy_macro_f1"]),
                int(round(float(current_proxy["val_proxy_total_false_count"]))),
                float(current_fullseq["val_pair_f1"]),
                "best" if improved else "keep-prev",
            )

            if ema_state is not None:
                current_state = _clone_state_dict(model.tcn)
                model.tcn.load_state_dict(ema_state)
                ema_halo = _evaluate_stage0_halo_model(model, val_segments, device=device)
                ema_fullseq = _evaluate_stage0_fullseq_model(model, val_segments, device=device)
                ema_proxy = _evaluate_stage0_threshold_proxy(model, val_segments, device=device)
                ema_payload = {
                    "epoch": int(epoch),
                    "source": "ema",
                    "halo16_eval": {"metrics": ema_halo, "timing_mae_ms": float(_compute_timing_mae_ms(ema_halo))},
                    "fullseq_eval": {
                        "metrics": ema_fullseq,
                        "timing_mae_ms": float(_compute_timing_mae_ms(ema_fullseq)),
                    },
                    "threshold_proxy_eval": ema_proxy,
                }
                epoch_row["val_ema"] = ema_payload
                improved = _is_better_stage0_eval(ema_payload, best_eval_payload)
                if improved:
                    best_eval_payload = ema_payload
                    best_eval_state = _clone_tensor_dict(ema_state)
                    best_state_head = _clone_state_dict(model.state_head)
                    best_epoch = int(epoch)
                    best_source = "ema"
                LOGGER.info(
                    "stage0 val epoch=%03d source=ema score=%.6f halo16=%.6f proxy=%.6f false=%d fullseq=%.6f %s",
                    epoch,
                    _stage0_selection_score(halo_metrics=ema_halo, proxy_metrics=ema_proxy),
                    float(ema_halo["val_pair_f1"]),
                    float(ema_proxy["val_proxy_macro_f1"]),
                    int(round(float(ema_proxy["val_proxy_total_false_count"]))),
                    float(ema_fullseq["val_pair_f1"]),
                    "best" if improved else "keep-prev",
                )
                model.tcn.load_state_dict(current_state)

        history.append(epoch_row)
        if time.monotonic() >= deadline:
            break

    if best_eval_state is None:
        model.eval()
        best_eval_payload = {
            "epoch": int(max(0, len(history) - 1)),
            "source": "current",
            "halo16_eval": {
                "metrics": _evaluate_stage0_halo_model(model, val_segments, device=device),
            },
            "fullseq_eval": {
                "metrics": _evaluate_stage0_fullseq_model(model, val_segments, device=device),
            },
            "threshold_proxy_eval": _evaluate_stage0_threshold_proxy(model, val_segments, device=device),
        }
        best_eval_state = _clone_state_dict(model.tcn)
        best_state_head = _clone_state_dict(model.state_head)
        best_epoch = int(max(0, len(history) - 1))
        best_source = "current"

    model.tcn.load_state_dict(best_eval_state)
    if best_state_head is not None:
        model.state_head.load_state_dict(best_state_head)
    final_halo_metrics = _evaluate_stage0_halo_model(model, val_segments, device=device)
    final_fullseq_metrics = _evaluate_stage0_fullseq_model(model, val_segments, device=device)
    final_proxy_metrics = _evaluate_stage0_threshold_proxy(model, val_segments, device=device)
    final_metrics = _stage0_selection_metrics(
        halo_metrics=final_halo_metrics,
        fullseq_metrics=final_fullseq_metrics,
        proxy_metrics=final_proxy_metrics,
    )
    final_checkpoint = output_dir / "boundary_model.pt"
    _save_stage0_checkpoint(
        final_checkpoint,
        model=model,
        cfg=cfg,
        metrics=final_metrics,
        best_epoch=best_epoch,
        best_source=best_source,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        train_records=train_records,
        val_records=val_records,
        pooler_path=spec.pooler_path,
        cache_root=spec.cache_root,
        state_class_weights=state_class_weights_np,
    )

    history_path = output_dir / "history.json"
    metrics_path = output_dir / "metrics.json"
    config_snapshot_path = output_dir / "config_snapshot.json"
    _write_json(
        history_path,
        {
            "history": history,
            "peak_vram_mb": float(peak_vram_mb),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
        },
    )
    _write_json(
        metrics_path,
        {
            "best_epoch": int(best_epoch),
            "best_source": str(best_source),
            "selection_metric": "guarded_avg_halo16_pair_f1_proxy_macro_f1_then_false_then_fullseq",
            "metrics": final_metrics,
            "halo16_metrics": final_halo_metrics,
            "fullseq_metrics": final_fullseq_metrics,
            "proxy_metrics": final_proxy_metrics,
            "timing_mae_ms": float(_compute_timing_mae_ms(final_halo_metrics)),
        },
    )
    _write_json(
        config_snapshot_path,
        {
            "stage0_config": asdict(cfg),
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps": int(total_steps),
            "time_budget_seconds": float(budget_seconds),
            "split_policy": str(spec.split_plan.split_policy),
            "val_ratio": float(VAL_RATIO),
            "halo": int(HALO_CONTEXT_WINDOWS),
            "center_chunk_len": int(HALO_CENTER_CHUNK),
            "state_names": list(STATE_NAMES),
            "state_class_weights": [float(item) for item in state_class_weights_np.tolist()],
            "state_aux_weight": float(cfg.aux_loss_weight),
            "shoulder_radius": int(STATE_SHOULDER_RADIUS),
            "shoulder_weight": float(STATE_SHOULDER_WEIGHT),
        },
    )
    return Stage0Result(
        checkpoint_path=final_checkpoint,
        config_snapshot_path=config_snapshot_path,
        metrics_path=metrics_path,
        history_path=history_path,
        metrics=final_metrics,
        halo_metrics=final_halo_metrics,
        fullseq_metrics=final_fullseq_metrics,
        proxy_metrics=final_proxy_metrics,
        best_epoch=int(best_epoch),
        best_source=str(best_source),
        steps_per_epoch=int(steps_per_epoch),
        total_steps=int(total_steps),
        elapsed_seconds=float(time.time() - start_t),
    )


def _resolve_vjepa_vendor_root() -> Path:
    vendor_root = (WORKSPACE_ROOT / "third_party" / "vjepa2_testing").resolve()
    if not vendor_root.exists():
        raise FileNotFoundError(f"Missing vendored V-JEPA root: {vendor_root}")
    return vendor_root


@contextmanager
def _use_vjepa_vendor_src_namespace():
    vendor_root = _resolve_vjepa_vendor_root()
    vendor_vjepa2 = (vendor_root / "vjepa2").resolve()
    autoresearch_root = str(ROOT.resolve())
    repo_root = str(WORKSPACE_ROOT)
    repo_src = str((WORKSPACE_ROOT / "src").resolve())
    vendor_root_str = str(vendor_root)
    vendor_vjepa2_str = str(vendor_vjepa2)
    saved_sys_path = list(sys.path)
    saved_src_modules = {
        key: module
        for key, module in list(sys.modules.items())
        if key == "src" or key.startswith("src.")
    }
    try:
        for key in list(saved_src_modules):
            sys.modules.pop(key, None)
        sys.path = [
            entry
            for entry in sys.path
            if entry not in {"", autoresearch_root, repo_root, repo_src, vendor_root_str, vendor_vjepa2_str}
        ]
        sys.path.insert(0, vendor_root_str)
        sys.path.insert(1, vendor_vjepa2_str)
        src_search_locations = [str(vendor_root / "src"), str(vendor_vjepa2 / "src")]
        src_pkg = types.ModuleType("src")
        src_pkg.__file__ = str(vendor_root / "src" / "constants.py")
        src_pkg.__package__ = "src"
        src_pkg.__path__ = list(src_search_locations)
        src_spec = importlib.machinery.ModuleSpec("src", loader=None, is_package=True)
        src_spec.submodule_search_locations = list(src_search_locations)
        src_pkg.__spec__ = src_spec
        sys.modules["src"] = src_pkg
        yield
    finally:
        sys.path = saved_sys_path
        for key in [k for k in list(sys.modules.keys()) if k == "src" or k.startswith("src.")]:
            sys.modules.pop(key, None)
        sys.modules.update(saved_src_modules)


@contextmanager
def _temporary_env(overrides: Mapping[str, Optional[str]]):
    saved = {key: os.environ.get(key) for key in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_probe_with_vendor_classifier(
    input_embed_dim: int,
    device: torch.device,
    pooler_path: Path,
    *,
    encoder_model: str,
    encoder_checkpoint: str,
) -> nn.Module:
    vendor_root = _resolve_vjepa_vendor_root()
    module_path = vendor_root / "src" / "pipeline" / "model_utils.py"
    module_name = f"_autoresearch_vjepa_model_utils_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load vendored model_utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    env_overrides = {
        "VJEPA_PROJECT_ROOT": str(vendor_root),
        "VJEPA_ENCODER_MODEL": str(encoder_model),
        "VJEPA_ENCODER_CHECKPOINT": str(encoder_checkpoint),
    }
    try:
        with _temporary_env(env_overrides), _use_vjepa_vendor_src_namespace():
            spec.loader.exec_module(module)
            build_classifier = getattr(module, "build_classifier")
            classifier = build_classifier(int(input_embed_dim), device)
    finally:
        sys.modules.pop(module_name, None)
    pooler_state = prod_probe._load_pooler_state_for_classifier(pooler_path)
    missing, unexpected = classifier.load_state_dict(pooler_state, strict=False)
    if missing or unexpected:
        LOGGER.info("Probe load warnings missing=%s unexpected=%s", missing[:8], unexpected[:8])
    return classifier.pooler.to(device)


def _resolve_left_context_from_tcn_payload(payload: Dict[str, Any]) -> int:
    model_cfg = payload.get("model_cfg", {}) if isinstance(payload, dict) else {}
    train_cfg = payload.get("train_cfg", {}) if isinstance(payload, dict) else {}
    seq_model = str(payload.get("seq_model", "")).strip().lower() if isinstance(payload, dict) else ""
    if seq_model == "hybrid_tcn_lstm":
        return 0
    left_ctx_raw = model_cfg.get("left_context") if isinstance(model_cfg, dict) else None
    if left_ctx_raw is not None:
        return int(max(0, int(left_ctx_raw)))
    bidirectional = False
    if isinstance(model_cfg, dict):
        bidirectional = bool(model_cfg.get("bidirectional") or model_cfg.get("tcn_bidirectional"))
    if not bidirectional and isinstance(train_cfg, dict):
        bidirectional = bool(train_cfg.get("bidirectional") or train_cfg.get("tcn_bidirectional"))
    if bidirectional:
        return 0
    dilations = model_cfg.get("dilations") if isinstance(model_cfg, dict) else None
    kernel_size = int(model_cfg.get("kernel_size", 3)) if isinstance(model_cfg, dict) else 3
    if isinstance(dilations, (list, tuple)) and dilations:
        return int((kernel_size - 1) * 2 * sum(int(item) for item in dilations))
    return 0


class HistoricalProbeTrainer:
    def __init__(
        self,
        spec: SpaceSpec,
        *,
        device: torch.device,
        tcn_checkpoint: Path,
        pooler_path: Path,
        encoder_model: str,
        encoder_checkpoint: Path,
        config: prod_probe.ProbePhase1Config,
    ) -> None:
        self.spec = spec
        self.device = device
        self.config = config
        self.current_pooler_path = Path(pooler_path).expanduser().resolve()
        self.current_tcn_path = Path(tcn_checkpoint).expanduser().resolve()
        self.history: List[Dict[str, float]] = []
        self.epoch = 0
        self._feature_cache: Dict[Path, Any] = {}

        self.tcn_payload = torch.load(self.current_tcn_path, map_location="cpu")
        if not isinstance(self.tcn_payload, dict):
            raise RuntimeError(f"Invalid boundary checkpoint payload: {self.current_tcn_path}")
        train_cfg_ckpt = self.tcn_payload.get("train_cfg", {}) if isinstance(self.tcn_payload.get("train_cfg"), dict) else {}
        self.boundary_index_mode = str(train_cfg_ckpt.get("boundary_index_mode", "ordered_nearest"))
        self.ignore_radius = int(train_cfg_ckpt.get("ignore_radius", 1))
        self.smooth_sigma = float(train_cfg_ckpt.get("smooth_sigma", 0.0))
        self.streams = [
            prod_probe._prepare_stream(
                name=str(record.segment_id),
                token_npz=Path(record.feature_path),
                labels_path=Path(record.label_path),
                ignore_radius=int(self.ignore_radius),
                smooth_sigma=float(self.smooth_sigma),
                boundary_index_mode=str(self.boundary_index_mode),
                feature_cache=self._feature_cache,
            )
            for record in self.spec.split_plan.train_records
        ]
        y_cycle_all = np.concatenate([stream.y_cycle[stream.mask_cycle > 0.5] for stream in self.streams], axis=0)
        self.pos_weight_cycle = float(
            prod_tcn._resolve_cycle_pos_weight(
                y_cycle_all,
                type(
                    "_Cfg",
                    (),
                    {
                        "pos_weight_cycle": 1.0,
                        "cycle_weight_trigger_low": 0.35,
                        "cycle_weight_trigger_high": 0.65,
                        "cycle_weight_min": 1.0,
                        "cycle_weight_max": 5.0,
                    },
                )(),
            )
        )
        self.boundary_loss_name = str(train_cfg_ckpt.get("boundary_loss", "focal")).lower()
        self.gamma = float(train_cfg_ckpt.get("gamma", 2.0))
        self.pos_weight_start_end = float(train_cfg_ckpt.get("pos_weight_start_end", 10.0))
        self.left_ctx = int(_resolve_left_context_from_tcn_payload(self.tcn_payload))

        self.probe = _build_probe_with_vendor_classifier(
            int(self.streams[0].tokens.shape[-1]),
            self.device,
            self.current_pooler_path,
            encoder_model=str(encoder_model),
            encoder_checkpoint=str(encoder_checkpoint),
        )
        self.tcn = prod_tcn.load_boundary_checkpoint(
            self.current_tcn_path,
            input_dim=int(self.streams[0].tokens.shape[-1]),
            device=str(self.device),
            expected_seq_model=str(self.tcn_payload.get("seq_model", "tcn")),
        ).to(self.device)
        prod_probe._set_probe_trainable(self.probe, "light")
        prod_probe._set_tcn_trainable(
            self.tcn,
            mode=str(self.config.tcn_tune_mode),
            last_blocks=int(self.config.tcn_last_blocks),
        )
        prod_probe._set_aux_sequence_trainable(self.tcn, enabled=True)
        self.optimizer, self.optimizer_impl = prod_probe._build_optimizer(
            probe=self.probe,
            tcn=self.tcn,
            config=self.config,
        )
        self.use_autocast = self.device.type == "cuda"
        self.autocast_dtype = torch.bfloat16
        self.pos_w_se_t = torch.tensor(float(self.pos_weight_start_end), device=self.device)
        self.pos_w_cycle_t = torch.tensor(float(self.pos_weight_cycle), device=self.device)


    def clone_probe_state(self) -> Dict[str, torch.Tensor]:
        return _clone_state_dict(self.probe)


    def clone_tcn_state(self) -> Dict[str, torch.Tensor]:
        return _clone_state_dict(self.tcn)


    def restore_states(
        self,
        *,
        probe_state: Dict[str, torch.Tensor],
        tcn_state: Dict[str, torch.Tensor],
    ) -> None:
        _restore_state_dict(self.probe, probe_state)
        _restore_state_dict(self.tcn, tcn_state)


    def train_epoch(self) -> Dict[str, float]:
        if int(self.epoch) < int(self.config.stage1_last_block_epoch):
            probe_mode = "light"
        elif int(self.epoch) < int(self.config.stage1_all_epoch):
            probe_mode = "last_blocks"
        else:
            probe_mode = "all"
        prod_probe._set_probe_trainable(self.probe, probe_mode)
        prod_probe._set_tcn_trainable(
            self.tcn,
            mode=str(self.config.tcn_tune_mode),
            last_blocks=int(self.config.tcn_last_blocks),
        )
        prod_probe._set_aux_sequence_trainable(self.tcn, enabled=True)
        self.probe.train()
        prod_probe._set_sequence_train_mode(self.tcn)

        rng = np.random.RandomState(int(self.config.seed) + int(self.epoch))
        step_queue = prod_probe._build_step_queue(
            self.streams,
            chunks_per_stream=int(self.config.chunks_per_stream),
            rng=rng,
            mode=str(self.config.stream_sampling_mode),
            power=float(self.config.stream_sampling_power),
            min_weight=float(self.config.stream_sampling_min_weight),
        )
        totals = {
            "loss": 0.0,
            "loss_boundary": 0.0,
            "loss_start": 0.0,
            "loss_end": 0.0,
            "loss_cycle": 0.0,
            "loss_cyclece": 0.0,
            "loss_smooth": 0.0,
            "loss_distill": 0.0,
        }
        steps = 0
        use_frozen_tcn_context = not prod_probe._has_trainable_tcn_branch(self.tcn)

        while step_queue:
            stream_idx = int(step_queue.pop())
            stream = self.streams[stream_idx]
            total_len = int(stream.tokens.shape[0])
            chunk_len = int(self.config.chunk_len if self.config.chunk_len > 0 else total_len)
            start = prod_tcn._select_chunk_start(
                rng=rng,
                T=total_len,
                chunk_len=chunk_len,
                boundary_indices=stream.boundary_indices,
                neg_chunk_fraction=float(self.config.neg_chunk_fraction),
            )
            end = min(total_len, int(start + chunk_len))
            ctx_start = max(0, int(start - self.left_ctx))
            offset = int(start - ctx_start)
            chunk_len_eff = int(end - start)
            if chunk_len_eff <= 0:
                continue

            y_cycle = torch.from_numpy(stream.y_cycle[start:end]).to(device=self.device)
            mask_cycle = torch.from_numpy(stream.mask_cycle[start:end]).to(device=self.device)
            mask_start_end = torch.from_numpy(stream.mask_start_end[start:end]).to(device=self.device)
            z0 = _torch_from_numpy_safe(stream.z0[start:end], device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            autocast_ctx = _autocast_context(enabled=bool(self.use_autocast), dtype=self.autocast_dtype)
            with autocast_ctx:
                if bool(use_frozen_tcn_context) and int(offset) > 0:
                    tokens_chunk = prod_probe._move_token_chunk_to_device(
                        _numpy_writable(stream.tokens[start:end]),
                        device=self.device,
                        use_autocast=bool(self.use_autocast),
                    )
                    z_ctx_prefix = _torch_from_numpy_safe(stream.z0[ctx_start:start], device=self.device)
                    z = self.probe(tokens_chunk).squeeze(1)
                    z_ctx = torch.cat([z_ctx_prefix.to(dtype=z.dtype), z], dim=0)
                    logits_win = self.tcn(z_ctx.unsqueeze(0)).squeeze(0)
                else:
                    tokens_win = prod_probe._move_token_chunk_to_device(
                        _numpy_writable(stream.tokens[ctx_start:end]),
                        device=self.device,
                        use_autocast=bool(self.use_autocast),
                    )
                    z_win = self.probe(tokens_win).squeeze(1)
                    logits_win = self.tcn(z_win.unsqueeze(0)).squeeze(0)
                    z = z_win[offset : offset + chunk_len_eff]
                logits = logits_win[offset : offset + chunk_len_eff]

            logits_start_end = logits[:, 0:2]
            logits_cycle = logits[:, 2]
            y_start = torch.from_numpy(stream.y_start[start:end]).to(device=self.device)
            y_end = torch.from_numpy(stream.y_end[start:end]).to(device=self.device)
            y_se = torch.stack([y_start, y_end], dim=-1)
            loss_se_raw = prod_tcn._boundary_loss_with_logits(
                logits=logits_start_end,
                targets=y_se,
                pos_weight=self.pos_w_se_t,
                gamma=float(self.gamma),
                loss_name=str(self.boundary_loss_name),
            )
            loss_start = prod_tcn.masked_mean(loss_se_raw[:, 0], mask_start_end)
            loss_end = prod_tcn.masked_mean(loss_se_raw[:, 1], mask_start_end)
            loss_boundary = loss_start + loss_end
            loss_cycle_raw = prod_tcn.bce_with_logits(
                logits_cycle,
                y_cycle,
                pos_weight=self.pos_w_cycle_t if float(self.pos_weight_cycle) != 1.0 else None,
                reduction="none",
            )
            loss_cycle = prod_tcn.masked_mean(loss_cycle_raw, mask_cycle)

            local_cycles = prod_probe._local_cycles_in_chunk(stream.mapped_cycles_idx, start=start, end=end)
            loss_cyclece = logits.new_tensor(0.0)
            if float(self.config.cyclece_weight) > 0.0 and local_cycles:
                loss_cyclece = prod_tcn._cycle_ce_loss(
                    logits=logits[:, :3],
                    mapped_cycles_idx=local_cycles,
                    combine=False,
                    tau=float(self.config.cyclece_tau),
                    radius=int(self.config.cyclece_radius),
                )

            loss_smooth = logits.new_tensor(0.0)
            if float(self.config.smooth_weight) > 0.0:
                loss_smooth = prod_probe._smooth_loss(z, mask_cycle)

            loss_distill = logits.new_tensor(0.0)
            if float(self.config.distill_weight) > 0.0:
                loss_distill = prod_probe._distill_loss(z, z0, mask_cycle)

            loss = (
                loss_boundary
                + loss_cycle
                + (loss_cyclece * float(self.config.cyclece_weight))
                + (loss_smooth * float(self.config.smooth_weight))
                + (loss_distill * float(self.config.distill_weight))
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected in probe phase1 space={self.spec.name}")
            loss.backward()
            if float(self.config.grad_clip_norm) > 0:
                params = [
                    param
                    for param in list(self.probe.parameters()) + list(self.tcn.parameters())
                    if param.requires_grad
                ]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, float(self.config.grad_clip_norm))
            self.optimizer.step()

            totals["loss"] += float(loss.item())
            totals["loss_boundary"] += float(loss_boundary.item())
            totals["loss_start"] += float(loss_start.item())
            totals["loss_end"] += float(loss_end.item())
            totals["loss_cycle"] += float(loss_cycle.item())
            totals["loss_cyclece"] += float(loss_cyclece.item())
            totals["loss_smooth"] += float(loss_smooth.item())
            totals["loss_distill"] += float(loss_distill.item())
            steps += 1

        if steps <= 0:
            raise RuntimeError(f"No optimization steps executed in probe phase1 space={self.spec.name}")

        row = {"epoch": float(self.epoch)}
        for key, value in totals.items():
            row[key] = float(value / float(steps))
        self.history.append(row)
        self.epoch += 1
        LOGGER.info(
            "Probe epoch=%d loss=%.4f probe_mode=%s trainable_probe_M=%.3f",
            self.epoch,
            row["loss"],
            probe_mode,
            _count_trainable_params(self.probe) / 1e6,
        )
        return row


    def _encode_tokens_full(self, tokens: np.ndarray) -> np.ndarray:
        outputs: List[np.ndarray] = []
        chunk = int(max(1, EVAL_TOKEN_CHUNK))
        self.probe.eval()
        autocast_ctx = _autocast_context(enabled=bool(self.use_autocast), dtype=self.autocast_dtype)
        for start in range(0, int(tokens.shape[0]), chunk):
            end = min(int(tokens.shape[0]), int(start + chunk))
            token_chunk = prod_probe._move_token_chunk_to_device(
                _numpy_writable(tokens[start:end]),
                device=self.device,
                use_autocast=bool(self.use_autocast),
            )
            with torch.no_grad():
                with autocast_ctx:
                    pooled = self.probe(token_chunk).squeeze(1)
            outputs.append(pooled.detach().float().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(outputs, axis=0)


    def _predict_pairs_for_record(self, record: SegmentRecord) -> List[EventPair]:
        payload = load_segment_arrays(record, representation="both", use_eval_span=True)
        pooled = self._encode_tokens_full(payload["tokens"])
        logits = _predict_halo_logits_from_pooled(
            self.tcn,
            pooled,
            device=self.device,
        ).astype(np.float32, copy=False)
        probs = 1.0 / (1.0 + np.exp(-logits[:, :3]))
        return decode_event_pairs(probs, payload["timestamps_ms"], heads=("start", "end", "cycle"))


    def evaluate(self) -> Dict[str, float]:
        self.probe.eval()
        self.tcn.eval()
        decode_heads, base_heads = _checkpoint_decode_heads(self.tcn_payload)
        eval_segments: List[ValidationEvalSegment] = []
        for record in self.spec.split_plan.val_eval_records:
            payload = load_segment_arrays(record, representation="both", use_eval_span=True)
            pooled = self._encode_tokens_full(payload["tokens"])
            logits = _predict_halo_logits_from_pooled(
                self.tcn,
                pooled,
                device=self.device,
            ).astype(np.float32, copy=False)
            eval_segments.append(
                ValidationEvalSegment(
                    record=record,
                    timestamps_ms=payload["timestamps_ms"],
                    logits=logits[:, :base_heads],
                    mapped_cycles_idx=_map_eval_cycles_idx(
                        record,
                        timestamps_ms=payload["timestamps_ms"],
                        boundary_index_mode=str(self.boundary_index_mode),
                    ),
                )
            )
        return _evaluate_validation_segments(eval_segments, decode_heads=decode_heads)


    def save_checkpoints(self, output_dir: Path) -> Tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe_state = {f"pooler.{key}": value.detach().cpu() for key, value in self.probe.state_dict().items()}
        pooler_sha = prod_probe._sha_from_state_dict(probe_state, salt=":probe_phase1")
        probe_ckpt = output_dir / "probe_pretrained.pt"
        torch.save(
            {
                "pooler_state": probe_state,
                "metadata": {
                    "pooler_sha": str(pooler_sha),
                    "encoder_model": str(self.spec.encoder_model),
                    "encoder_checkpoint": str(self.spec.encoder_checkpoint),
                    "source": "dense_temporal_probe_phase1",
                    "stage": "probe_phase1",
                },
            },
            probe_ckpt,
        )

        tcn_payload = dict(self.tcn_payload)
        tcn_payload["model_state"] = {key: value.detach().cpu() for key, value in self.tcn.state_dict().items()}
        tcn_payload["probe_finetune"] = {
            "stage": "probe_phase1",
            "temporal_structure_mode": "cyclic",
            "effective_losses": {
                "boundary": True,
                "class": False,
                "cycle": True,
                "phase": False,
                "ranking": False,
                "cyclece": float(self.config.cyclece_weight) > 0.0,
                "smooth": float(self.config.smooth_weight) > 0.0,
                "distill": float(self.config.distill_weight) > 0.0,
            },
            "best_epoch": int(max(0, self.epoch - 1)),
            "best_loss": float(min((row["loss"] for row in self.history), default=float("inf"))),
            "cyclece_weight": float(self.config.cyclece_weight),
            "smooth_weight": float(self.config.smooth_weight),
            "distill_weight": float(self.config.distill_weight),
            "class_weight": 0.0,
        }
        tcn_ckpt = output_dir / "boundary_model.pt"
        torch.save(tcn_payload, tcn_ckpt)
        (output_dir / "train_metrics.json").write_text(
            json.dumps({"history": self.history}, indent=2),
            encoding="utf-8",
        )
        return probe_ckpt, tcn_ckpt


def _historical_phase1_config(*, seed: int, chunk_len: int) -> prod_probe.ProbePhase1Config:
    return prod_probe.ProbePhase1Config(
        epochs=20,
        probe_lr=2.6e-5,
        weight_decay=1e-4,
        grad_clip_norm=1.0,
        chunk_len=int(chunk_len),
        chunks_per_stream=12,
        neg_chunk_fraction=0.25,
        stage1_last_block_epoch=0,
        stage1_all_epoch=999,
        tcn_tune_mode="frozen",
        tcn_last_blocks=1,
        tcn_lr=5e-5,
        tcn_weight_decay=1e-4,
        stream_sampling_mode="uniform",
        stream_sampling_power=0.5,
        stream_sampling_min_weight=1.0,
        cyclece_weight=1.0,
        cyclece_tau=0.7,
        cyclece_radius=1,
        smooth_weight=0.05,
        distill_weight=0.10,
        class_weight=0.0,
        fail_if_best_epoch_zero=False,
        boundary_index_mode="ordered_nearest",
        temporal_structure_mode="cyclic",
        seed=int(seed),
    )


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "cuda out of memory" in msg or "outofmemoryerror" in type(exc).__name__.lower()


def _run_historical_phase1(
    *,
    spec: SpaceSpec,
    stage0_result: Stage0Result,
    output_dir: Path,
    device: torch.device,
    budget_seconds: float,
) -> Phase1Result:
    output_dir.mkdir(parents=True, exist_ok=True)
    start_t = time.time()
    if float(budget_seconds) <= 0.0:
        history_path = output_dir / "history.json"
        metrics_path = output_dir / "metrics.json"
        config_snapshot_path = output_dir / "config_snapshot.json"
        _write_json(history_path, {"history": []})
        _write_json(
            metrics_path,
            {
                "skipped": True,
                "skip_reason": "zero_budget",
                "selection_metric": "stage0_selected_checkpoint_passthrough",
                "metrics": stage0_result.metrics,
                "source_stage0_checkpoint": str(stage0_result.checkpoint_path),
                "source_pooler_checkpoint": str(spec.pooler_path),
            },
        )
        _write_json(
            config_snapshot_path,
            {
                "profile_name": "historical",
                "skipped": True,
                "skip_reason": "zero_budget",
                "time_budget_seconds": float(budget_seconds),
            },
        )
        return Phase1Result(
            probe_checkpoint=spec.pooler_path,
            tcn_checkpoint=stage0_result.checkpoint_path,
            metrics_path=metrics_path,
            config_snapshot_path=config_snapshot_path,
            history_path=history_path,
            metrics=dict(stage0_result.metrics),
            best_epoch=int(stage0_result.best_epoch),
            best_loss=0.0,
            effective_chunk_len=0,
            elapsed_seconds=float(time.time() - start_t),
            skipped=True,
        )

    chunk_len = 32 if str(spec.encoder_model).lower() == "large" else 256
    while True:
        config = _historical_phase1_config(seed=SEED, chunk_len=chunk_len)
        LOGGER.info(
            "Phase-1 historical: epochs=%d chunk_len=%d tcn_tune_mode=%s stream_sampling=%s budget=%.1fs",
            int(config.epochs),
            int(config.chunk_len),
            str(config.tcn_tune_mode),
            str(config.stream_sampling_mode),
            float(budget_seconds),
        )
        trainer = HistoricalProbeTrainer(
            spec,
            device=device,
            tcn_checkpoint=stage0_result.checkpoint_path,
            pooler_path=spec.pooler_path,
            encoder_model=str(spec.encoder_model),
            encoder_checkpoint=spec.encoder_checkpoint,
            config=config,
        )
        best_loss = float("inf")
        best_epoch = -1
        best_probe_state: Optional[Dict[str, torch.Tensor]] = None
        best_tcn_state: Optional[Dict[str, torch.Tensor]] = None
        deadline = time.monotonic() + float(budget_seconds)
        try:
            for _ in range(int(config.epochs)):
                if trainer.epoch > 0 and time.monotonic() >= deadline:
                    break
                row = trainer.train_epoch()
                if float(row["loss"]) < float(best_loss):
                    best_loss = float(row["loss"])
                    best_epoch = int(trainer.epoch - 1)
                    best_probe_state = trainer.clone_probe_state()
                    best_tcn_state = trainer.clone_tcn_state()
                if time.monotonic() >= deadline:
                    break
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            next_chunk_len = max(int(PHASE1_MIN_CHUNK), int(chunk_len // 2))
            if next_chunk_len >= int(chunk_len):
                raise
            LOGGER.warning(
                "Historical phase-1 hit CUDA OOM at chunk_len=%d; retrying with chunk_len=%d",
                int(chunk_len),
                int(next_chunk_len),
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            chunk_len = int(next_chunk_len)
            continue

        if best_probe_state is None or best_tcn_state is None:
            best_probe_state = trainer.clone_probe_state()
            best_tcn_state = trainer.clone_tcn_state()
            best_epoch = int(max(0, trainer.epoch - 1))
            best_loss = float(trainer.history[-1]["loss"]) if trainer.history else float("inf")

        trainer.restore_states(probe_state=best_probe_state, tcn_state=best_tcn_state)
        metrics = trainer.evaluate()
        probe_checkpoint, tcn_checkpoint = trainer.save_checkpoints(output_dir)
        history_path = output_dir / "history.json"
        metrics_path = output_dir / "metrics.json"
        config_snapshot_path = output_dir / "config_snapshot.json"
        _write_json(history_path, {"history": trainer.history})
        _write_json(
            metrics_path,
            {
                "best_epoch": int(best_epoch),
                "best_loss": float(best_loss),
                "effective_chunk_len": int(chunk_len),
                "selection_metric": "threshold_proxy_macro_f1_then_false_count_then_legacy_pair_f1_then_count_mae_then_timing",
                "metrics": metrics,
                "timing_mae_ms": float(_compute_timing_mae_ms(metrics)),
            },
        )
        _write_json(
            config_snapshot_path,
            {
                "profile_name": "historical",
                "effective_config": asdict(config),
                "effective_chunk_len": int(chunk_len),
                "time_budget_seconds": float(budget_seconds),
            },
        )
        return Phase1Result(
            probe_checkpoint=probe_checkpoint,
            tcn_checkpoint=tcn_checkpoint,
            metrics_path=metrics_path,
            config_snapshot_path=config_snapshot_path,
            history_path=history_path,
            metrics=metrics,
            best_epoch=int(best_epoch),
            best_loss=float(best_loss),
            effective_chunk_len=int(chunk_len),
            elapsed_seconds=float(time.time() - start_t),
            skipped=False,
        )


def _print_split_summary(spec: SpaceSpec) -> None:
    singleton_cameras = sum(1 for count in spec.split_plan.camera_total_counts.values() if int(count) == 1)
    planned_lines = [
        f"{camera_id}:{spec.split_plan.camera_val_counts[camera_id]}/{spec.split_plan.camera_total_counts[camera_id]}"
        for camera_id in sorted(spec.split_plan.camera_total_counts)
    ]
    print(
        f"[split] {spec.name} | policy={spec.split_plan.split_policy} "
        f"train_videos={len(spec.split_plan.train_videos)} "
        f"val_videos={len(spec.split_plan.val_videos)} "
        f"target_val_videos={spec.split_plan.target_val_videos:.1f} "
        f"singleton_cameras={singleton_cameras}"
    )
    print(f"[split] {spec.name} camera_val_counts={' '.join(planned_lines)}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _set_seed(SEED)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_budget_seconds = resolve_time_budget_seconds()
    total_timeout_seconds = resolve_total_timeout_seconds()
    tcn_stage_seconds, probe_stage_seconds = resolve_stage_budget_seconds(total_budget_seconds)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = _resolve_output_root() / f"minda_subassembly_replay_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    spec = _build_space_spec(output_root)
    _print_split_summary(spec)

    print(f"[task] {TASK_MODE}")
    print(f"[workspace_root] {WORKSPACE_ROOT}")
    print(f"[device] {device}")
    print(f"[output_root] {output_root}")
    print(f"[source_run_dir] {spec.source_run_dir}")
    print(f"[cache_root] {spec.cache_root}")

    start_time = time.time()
    peak_vram_mb = 0.0
    stage0_cfg = Stage0Config(seed=SEED)

    print(
        f"[stage] tcn | duration={tcn_stage_seconds:.1f}s "
        f"| trainable_params_M={_count_trainable_params(Stage0BoundaryModel(input_dim=int(spec.split_plan.train_records[0].embedding_dim), cfg=stage0_cfg)) / 1e6:.3f}"
    )
    stage0_result = _run_stage0(
        spec=spec,
        output_dir=output_root / "stage0_best",
        cfg=stage0_cfg,
        budget_seconds=tcn_stage_seconds,
        device=device,
    )
    peak_vram_mb = max(peak_vram_mb, _peak_vram_mb())
    print(
        f"[tcn] best_epoch={stage0_result.best_epoch} source={stage0_result.best_source} "
        f"score={stage0_result.metrics['val_selection_score']:.6f} "
        f"halo16={stage0_result.metrics['val_halo16_pair_f1']:.6f} "
        f"proxy={stage0_result.metrics['val_proxy_macro_f1']:.6f} "
        f"false={int(round(stage0_result.metrics['val_proxy_total_false_count']))} "
        f"fullseq={stage0_result.metrics['val_fullseq_pair_f1']:.6f}"
    )

    print("[stage] probe_phase1 | duration=" f"{probe_stage_seconds:.1f}s | profile=historical")
    phase1_result = _run_historical_phase1(
        spec=spec,
        stage0_result=stage0_result,
        output_dir=output_root / "phase1_historical",
        device=device,
        budget_seconds=probe_stage_seconds,
    )
    peak_vram_mb = max(peak_vram_mb, _peak_vram_mb())
    if phase1_result.skipped:
        print(
            f"[probe] skipped=1 reason=zero_budget "
            f"passthrough_tcn={phase1_result.tcn_checkpoint} "
            f"passthrough_probe={phase1_result.probe_checkpoint}"
        )
    else:
        print(
            f"[probe] best_epoch={phase1_result.best_epoch} loss={phase1_result.best_loss:.6f} "
            f"pair_f1={phase1_result.metrics['val_pair_f1']:.6f} "
            f"legacy_pair_f1={_legacy_pair_f1(phase1_result.metrics):.6f} "
            f"count_mae={phase1_result.metrics['val_count_mae']:.6f} "
            f"chunk_len={phase1_result.effective_chunk_len}"
        )

    total_seconds = float(time.time() - start_time)
    summary = {
        "model_family": MODEL_FAMILY,
        "task_mode": TASK_MODE,
        "time_budget_seconds": float(total_budget_seconds),
        "tcn_stage_seconds": float(tcn_stage_seconds),
        "probe_stage_seconds": float(probe_stage_seconds),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "cache_root": str(spec.cache_root),
        "source_run_dir": str(spec.source_run_dir),
        "split_policy": str(spec.split_plan.split_policy),
        "train_segments": len(spec.split_plan.train_records),
        "val_segments": len(spec.split_plan.val_records),
        "val_eval_segments": len(spec.split_plan.val_eval_records),
        "train_videos": len(spec.split_plan.train_videos),
        "val_videos": len(spec.split_plan.val_videos),
        "stage0": {
            "checkpoint": str(stage0_result.checkpoint_path),
            "metrics_path": str(stage0_result.metrics_path),
            "config_snapshot_path": str(stage0_result.config_snapshot_path),
            "history_path": str(stage0_result.history_path),
            "best_epoch": int(stage0_result.best_epoch),
            "best_source": str(stage0_result.best_source),
            "elapsed_seconds": float(stage0_result.elapsed_seconds),
            "metrics": stage0_result.metrics,
            "halo16_metrics": stage0_result.halo_metrics,
            "fullseq_metrics": stage0_result.fullseq_metrics,
            "proxy_metrics": stage0_result.proxy_metrics,
        },
        "phase1": {
            "probe_checkpoint": str(phase1_result.probe_checkpoint),
            "tcn_checkpoint": str(phase1_result.tcn_checkpoint),
            "metrics_path": str(phase1_result.metrics_path),
            "config_snapshot_path": str(phase1_result.config_snapshot_path),
            "history_path": str(phase1_result.history_path),
            "best_epoch": int(phase1_result.best_epoch),
            "best_loss": float(phase1_result.best_loss),
            "effective_chunk_len": int(phase1_result.effective_chunk_len),
            "elapsed_seconds": float(phase1_result.elapsed_seconds),
            "skipped": bool(phase1_result.skipped),
            "metrics": phase1_result.metrics,
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if total_seconds > float(total_timeout_seconds):
        print("FAIL: exceeded total timeout")
        return 1

    print("---")
    if phase1_result.skipped:
        print(f"val_selection_score:{stage0_result.metrics['val_selection_score']:.6f}")
        print(f"val_halo16_pair_f1:{stage0_result.metrics['val_halo16_pair_f1']:.6f}")
        print(f"val_proxy_macro_f1:{stage0_result.metrics['val_proxy_macro_f1']:.6f}")
        print(f"val_proxy_false:   {int(round(stage0_result.metrics['val_proxy_total_false_count']))}")
        print(f"val_fullseq_pair_f1:{stage0_result.metrics['val_fullseq_pair_f1']:.6f}")
    else:
        print(f"val_pair_f1:        {phase1_result.metrics['val_pair_f1']:.6f}")
        print(f"val_legacy_pair_f1: {phase1_result.metrics['val_legacy_pair_f1']:.6f}")
        print(f"val_count_mae:      {phase1_result.metrics['val_count_mae']:.6f}")
        print(f"val_start_mae_ms:   {phase1_result.metrics['val_start_mae_ms']:.1f}")
        print(f"val_end_mae_ms:     {phase1_result.metrics['val_end_mae_ms']:.1f}")
    print(f"training_seconds:   {tcn_stage_seconds + probe_stage_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"time_budget_seconds:{total_budget_seconds:.1f}")
    print(f"tcn_stage_seconds:  {tcn_stage_seconds:.1f}")
    print(f"probe_stage_seconds:{probe_stage_seconds:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"cache_version:      {CACHE_VERSION}")
    print(f"model_family:       {MODEL_FAMILY}")
    print(f"task_mode:          {TASK_MODE}")
    print(f"pooler_tune_mode:   {'phase1_skipped_zero_budget' if phase1_result.skipped else 'phase1_historical'}")
    print(f"representation_mode:{'hybrid_halo16_pooled_z0' if phase1_result.skipped else 'pooled_z0_then_tokens'}")
    print(f"cache_root:         {spec.cache_root}")
    print(f"output_root_final:  {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
