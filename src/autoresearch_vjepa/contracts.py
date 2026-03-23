"""Dense temporal run-contract parsing for production training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _normalize_range(raw: Sequence[Any]) -> Optional[Tuple[int, int]]:
    if len(raw) < 2:
        return None
    s = int(raw[0])
    e = int(raw[1])
    if e < s:
        return None
    return int(s), int(e)


def _normalize_label(label: Any) -> str:
    token = str(label or "").strip().lower()
    if token in {"1", "action"}:
        return "action"
    if token in {"2", "ignore"}:
        return "ignore"
    return "idle"


def _normalize_label_name(label: Any, *, fallback: str) -> str:
    token = str(label or "").strip().lower().replace(" ", "_")
    token = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in token)
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        token = str(fallback or "").strip().lower()
    return token or str(fallback)


def _normalize_boundary_index_mode(value: Any) -> str:
    token = str(value or "legacy").strip().lower().replace("-", "_")
    if token in {"legacy", "nearest", "ordered_nearest"}:
        return token
    return "legacy"


def _normalize_temporal_structure_mode(value: Any) -> str:
    token = str(value or "cyclic").strip().lower().replace("-", "_")
    if token in {"cyclic", "event"}:
        return token
    return "cyclic"


@dataclass(frozen=True)
class DenseActionLabel:
    label_name: str
    label_id: int
    action_class_id: Optional[str] = None


@dataclass(frozen=True)
class DenseTemporalTargets:
    window_size: int
    window_stride: int
    version: str = "dense_v2"
    action_label: int = 1
    ignore_label: int = 2
    action_labels: Tuple[DenseActionLabel, ...] = field(default_factory=tuple)
    label_map: Dict[str, int] = field(default_factory=lambda: {"idle": 0, "action": 1, "ignore": 2})


@dataclass(frozen=True)
class DenseVideo:
    video_id: str
    path: str
    camera_id: str
    is_for_test: bool
    num_frames: Optional[int]
    fps: Optional[float]
    roi: Optional[Dict[str, float]]
    cycles: Tuple[Tuple[int, int], ...]
    ignore_regions: Tuple[Tuple[int, int], ...]
    regions: Tuple[Tuple[str, str, Optional[str], int, int], ...]
    frame_labels_rle: Tuple[Tuple[int, int, int], ...]

    @staticmethod
    def _parse_roi(raw: Any) -> Optional[Dict[str, float]]:
        if not isinstance(raw, dict):
            return None
        x = _as_float(raw.get("x"), 0.0)
        y = _as_float(raw.get("y"), 0.0)
        w = _as_float(raw.get("w", raw.get("width")), 1.0)
        h = _as_float(raw.get("h", raw.get("height")), 1.0)
        if w <= 0.0 or h <= 0.0:
            return None
        return {"x": x, "y": y, "w": w, "h": h}

    @classmethod
    def from_dict(cls, raw: Dict[str, Any], fallback_camera_id: str) -> "DenseVideo":
        video_id = str(raw.get("video_id") or Path(str(raw.get("path") or "")).stem or "unknown_video")
        camera_id = str(raw.get("camera_id") or fallback_camera_id or "dense_temporal_camera")
        cycles: List[Tuple[int, int]] = []
        for item in raw.get("cycles", []) or []:
            if isinstance(item, (list, tuple)):
                parsed = _normalize_range(item)
                if parsed is not None:
                    cycles.append(parsed)

        ignore_regions: List[Tuple[int, int]] = []
        for item in raw.get("ignore_regions", []) or []:
            if isinstance(item, (list, tuple)):
                parsed = _normalize_range(item)
                if parsed is not None:
                    ignore_regions.append(parsed)

        regions: List[Tuple[str, str, Optional[str], int, int]] = []
        for item in raw.get("regions", []) or []:
            if not isinstance(item, dict):
                continue
            start = item.get("start_frame")
            end = item.get("end_frame")
            if start is None or end is None:
                continue
            parsed = _normalize_range((start, end))
            if parsed is None:
                continue
            label_type = _normalize_label(item.get("label_type", item.get("label")))
            if label_type == "action":
                fallback_label_name = "action"
            elif label_type == "ignore":
                fallback_label_name = "ignore"
            else:
                fallback_label_name = "idle"
            label_name = _normalize_label_name(item.get("label_name"), fallback=fallback_label_name)
            action_class_id_raw = item.get("action_class_id")
            action_class_id = str(action_class_id_raw).strip() if action_class_id_raw is not None else ""
            regions.append(
                (
                    str(label_type),
                    str(label_name),
                    (action_class_id or None),
                    int(parsed[0]),
                    int(parsed[1]),
                )
            )

        rle: List[Tuple[int, int, int]] = []
        for item in raw.get("frame_labels_rle", []) or []:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            label = int(item[0])
            parsed = _normalize_range((item[1], item[2]))
            if parsed is None:
                continue
            rle.append((label, int(parsed[0]), int(parsed[1])))

        return cls(
            video_id=video_id,
            path=str(raw.get("path") or ""),
            camera_id=camera_id,
            is_for_test=_as_bool(raw.get("is_for_test"), False),
            num_frames=(_as_int(raw.get("num_frames"), -1) if raw.get("num_frames") is not None else None),
            fps=(_as_float(raw.get("fps"), 0.0) if raw.get("fps") is not None else None),
            roi=cls._parse_roi(raw.get("roi")),
            cycles=tuple(sorted(cycles)),
            ignore_regions=tuple(sorted(ignore_regions)),
            regions=tuple(regions),
            frame_labels_rle=tuple(rle),
        )


@dataclass(frozen=True)
class DenseTemporalModelSpec:
    pooler_path: str
    encoder_checkpoint: str
    encoder_model: str
    preproc_id: str
    inference_dtype: str
    device: str
    decode_device: str
    decode_gpu_id: int
    batch_size: int


@dataclass(frozen=True)
class DenseTemporalTrainSpec:
    clip_len: int
    stride: int
    frame_skip: int
    seq_model: str
    hidden_dim: int
    kernel_size: int
    dropout: float
    use_layernorm: bool
    dilations: Tuple[int, ...]
    tcn_bidirectional: bool
    tcn_task_specific_heads: bool
    use_phase_head: bool
    phase_loss_weight: float
    ranking_loss_weight: float
    cyclece_loss_weight: float
    cyclece_tau: float
    cyclece_radius: int
    class_loss_weight: float
    class_sampling_alpha: float
    ignore_radius: int
    smooth_sigma: float
    combine_start_end: bool
    boundary_index_mode: str
    boundary_loss: str
    gamma: float
    pos_weight_start_end: float
    pos_weight_cycle: Optional[float]
    lr: float
    weight_decay: float
    max_epochs: int
    chunk_len: int
    chunks_per_epoch: int
    neg_chunk_fraction: float
    grad_clip_norm: float
    stage1_enabled: bool
    stage1_epochs: int
    stage1_probe_lr: float
    stage1_last_block_epoch: int
    stage1_all_epoch: int
    stage1_chunks_per_stream: int
    stage1_neg_chunk_fraction: float
    stage1_tcn_tune_mode: str
    stage1_tcn_last_blocks: int
    stage1_tcn_lr: float
    stage1_tcn_weight_decay: float
    stage1_stream_sampling_mode: str
    stage1_stream_sampling_power: float
    stage1_stream_sampling_min_weight: float
    stage1_cyclece_weight: float
    stage1_smooth_weight: float
    stage1_distill_weight: float
    stage1_class_weight: float
    cyclecl_enabled: bool
    cyclecl_epochs: int
    cyclecl_patience_epochs: int
    cyclecl_max_cycles_per_video: int
    cyclecl_cycle_selection_mode: str
    cyclecl_min_cycle_frames: int
    cyclecl_boundary_tolerance_frames: int
    cyclecl_interior_margin_frames: int
    cyclecl_mine_batch_size: int
    cyclecl_triplet_batch_size: int
    cyclecl_max_triplets_per_epoch: int
    cyclecl_margin: float
    cyclecl_pooler_lr: float
    cyclecl_weight_decay: float
    cyclecl_hard_neg_topk: int
    cyclecl_pos_topk: int
    cyclecl_sampling_strategy: str
    cyclecl_mean_threshold_beta: float
    cyclecl_grad_clip_norm: float
    stage1_fail_if_best_epoch_zero: bool
    seed: int


@dataclass(frozen=True)
class DenseTemporalRunConfig:
    run_id: str
    space_id: str
    run_number: int
    dataset_mode: str
    temporal_structure_mode: str
    videos: Tuple[DenseVideo, ...]
    temporal_targets: DenseTemporalTargets
    model: DenseTemporalModelSpec
    train: DenseTemporalTrainSpec


def parse_dense_temporal_snapshot(
    *,
    run_id: str,
    space_id: str,
    run_number: int,
    snapshot: Dict[str, Any],
    env: Dict[str, str],
) -> DenseTemporalRunConfig:
    if not isinstance(snapshot, dict):
        raise TypeError("config_snapshot must be a dictionary")

    dataset_mode = str(snapshot.get("dataset_mode") or "").strip().lower()
    if dataset_mode != "dense":
        raise ValueError(f"dense_temporal requires dataset_mode='dense' (got {dataset_mode!r})")
    temporal_structure_mode = _normalize_temporal_structure_mode(snapshot.get("temporal_structure_mode"))

    temporal_targets_raw = snapshot.get("temporal_targets") if isinstance(snapshot.get("temporal_targets"), dict) else {}
    window_size = _as_int(temporal_targets_raw.get("window_size"), 4)
    window_stride = _as_int(temporal_targets_raw.get("window_stride"), 2)
    raw_label_map = temporal_targets_raw.get("label_map") if isinstance(temporal_targets_raw.get("label_map"), dict) else {}
    label_map: Dict[str, int] = {}
    for key, value in raw_label_map.items():
        name = str(key or "").strip().lower()
        if not name:
            continue
        try:
            label_map[name] = int(value)
        except (TypeError, ValueError):
            continue
    if not label_map:
        label_map = {"idle": 0, "action": 1, "ignore": 2}

    action_labels_list: List[DenseActionLabel] = []
    raw_action_labels = temporal_targets_raw.get("action_labels")
    if isinstance(raw_action_labels, list):
        for item in raw_action_labels:
            if not isinstance(item, dict):
                continue
            label_name = _normalize_label_name(item.get("label_name"), fallback="action")
            label_id = _as_int(item.get("label_id"), label_map.get(label_name, -1))
            if label_id < 0:
                continue
            action_class_id_raw = item.get("action_class_id")
            action_class_id = str(action_class_id_raw).strip() if action_class_id_raw is not None else ""
            action_labels_list.append(
                DenseActionLabel(
                    label_name=str(label_name),
                    label_id=int(label_id),
                    action_class_id=(action_class_id or None),
                )
            )

    if not action_labels_list:
        action_label_fallback = _as_int(temporal_targets_raw.get("action_label"), label_map.get("action", 1))
        fallback_action_name = "action"
        for label_name, label_id in label_map.items():
            if int(label_id) == int(action_label_fallback):
                fallback_action_name = _normalize_label_name(label_name, fallback="action")
                break
        action_labels_list = [
            DenseActionLabel(
                label_name=str(fallback_action_name),
                label_id=int(action_label_fallback),
                action_class_id=None,
            )
        ]

    # Deduplicate by label_name while preserving order.
    deduped_action_labels: List[DenseActionLabel] = []
    seen_action_names: set[str] = set()
    for action_label in action_labels_list:
        if action_label.label_name in seen_action_names:
            continue
        seen_action_names.add(action_label.label_name)
        deduped_action_labels.append(action_label)

    action_label = _as_int(
        temporal_targets_raw.get("action_label"),
        deduped_action_labels[0].label_id if deduped_action_labels else label_map.get("action", 1),
    )
    ignore_label = _as_int(temporal_targets_raw.get("ignore_label"), label_map.get("ignore", 2))
    temporal_targets = DenseTemporalTargets(
        window_size=max(1, int(window_size)),
        window_stride=max(1, int(window_stride)),
        version=str(temporal_targets_raw.get("version") or "dense_v2").strip().lower(),
        action_label=int(action_label),
        ignore_label=int(ignore_label),
        action_labels=tuple(deduped_action_labels),
        label_map=dict(label_map),
    )

    videos_raw = snapshot.get("videos") if isinstance(snapshot.get("videos"), list) else []
    if not videos_raw:
        raise ValueError("dense_temporal requires non-empty videos[] in config_snapshot")
    fallback_camera = str(snapshot.get("camera_id") or "dense_temporal_camera")
    videos = tuple(DenseVideo.from_dict(v, fallback_camera) for v in videos_raw if isinstance(v, dict))
    if not videos:
        raise ValueError("dense_temporal could not parse any videos from config_snapshot")

    model_raw = snapshot.get("temporal_model") if isinstance(snapshot.get("temporal_model"), dict) else {}
    pooler_path = str(model_raw.get("pooler_path") or env.get("DENSE_TEMPORAL_POOLER_PATH") or "").strip()
    encoder_checkpoint = str(model_raw.get("encoder_checkpoint") or env.get("DENSE_TEMPORAL_ENCODER_CHECKPOINT") or "").strip()
    if not pooler_path:
        raise ValueError("Missing temporal_model.pooler_path (or DENSE_TEMPORAL_POOLER_PATH)")
    if not encoder_checkpoint:
        raise ValueError("Missing temporal_model.encoder_checkpoint (or DENSE_TEMPORAL_ENCODER_CHECKPOINT)")

    model = DenseTemporalModelSpec(
        pooler_path=pooler_path,
        encoder_checkpoint=encoder_checkpoint,
        encoder_model=str(model_raw.get("encoder_model") or env.get("DENSE_TEMPORAL_ENCODER_MODEL") or "large").strip().lower(),
        preproc_id=str(model_raw.get("preproc_id") or env.get("DENSE_TEMPORAL_PREPROC_ID") or "vjepa_rgb_256").strip(),
        inference_dtype=str(model_raw.get("inference_dtype") or env.get("DENSE_TEMPORAL_INFERENCE_DTYPE") or "bf16").strip().lower(),
        device=str(model_raw.get("device") or env.get("DENSE_TEMPORAL_DEVICE") or "cuda").strip(),
        decode_device=str(model_raw.get("decode_device") or env.get("DENSE_TEMPORAL_DECODE_DEVICE") or "gpu").strip().lower(),
        decode_gpu_id=_as_int(model_raw.get("decode_gpu_id", env.get("DENSE_TEMPORAL_DECODE_GPU_ID")), 0),
        batch_size=max(1, _as_int(model_raw.get("batch_size", env.get("DENSE_TEMPORAL_BATCH_SIZE")), 64)),
    )

    train_raw = snapshot.get("temporal_training") if isinstance(snapshot.get("temporal_training"), dict) else {}
    clip_len = max(1, _as_int(train_raw.get("clip_len", env.get("DENSE_TEMPORAL_CLIP_LEN")), temporal_targets.window_size))
    stride = max(1, _as_int(train_raw.get("stride", env.get("DENSE_TEMPORAL_STRIDE")), temporal_targets.window_stride))
    dilations_raw = train_raw.get("dilations", env.get("DENSE_TEMPORAL_DILATIONS"))
    dilations: Tuple[int, ...]
    if isinstance(dilations_raw, str):
        parts = [p.strip() for p in dilations_raw.split(",") if p.strip()]
        dilations = tuple(max(1, int(p)) for p in parts) if parts else (1, 2, 4, 8, 16, 32)
    elif isinstance(dilations_raw, (list, tuple)):
        dilations = tuple(max(1, int(x)) for x in dilations_raw) or (1, 2, 4, 8, 16, 32)
    else:
        dilations = (1, 2, 4, 8, 16, 32)

    pos_weight_cycle_raw = train_raw.get("pos_weight_cycle", env.get("DENSE_TEMPORAL_POS_WEIGHT_CYCLE"))
    pos_weight_cycle = None
    if pos_weight_cycle_raw is not None and str(pos_weight_cycle_raw).strip() != "":
        pos_weight_cycle = float(pos_weight_cycle_raw)

    train = DenseTemporalTrainSpec(
        clip_len=clip_len,
        stride=stride,
        frame_skip=max(0, _as_int(train_raw.get("frame_skip", env.get("DENSE_TEMPORAL_FRAME_SKIP")), 0)),
        seq_model=str(train_raw.get("seq_model", env.get("DENSE_TEMPORAL_SEQ_MODEL") or "tcn")).strip().lower(),
        hidden_dim=max(8, _as_int(train_raw.get("hidden_dim", env.get("DENSE_TEMPORAL_HIDDEN_DIM")), 128)),
        kernel_size=max(2, _as_int(train_raw.get("kernel_size", env.get("DENSE_TEMPORAL_KERNEL_SIZE")), 5)),
        dropout=max(0.0, _as_float(train_raw.get("dropout", env.get("DENSE_TEMPORAL_DROPOUT")), 0.1)),
        use_layernorm=_as_bool(train_raw.get("use_layernorm", env.get("DENSE_TEMPORAL_USE_LAYERNORM")), True),
        dilations=dilations,
        tcn_bidirectional=_as_bool(
            train_raw.get("tcn_bidirectional", env.get("DENSE_TEMPORAL_TCN_BIDIRECTIONAL")),
            True,
        ),
        tcn_task_specific_heads=_as_bool(
            train_raw.get("tcn_task_specific_heads", env.get("DENSE_TEMPORAL_TCN_TASK_SPECIFIC_HEADS")),
            True,
        ),
        use_phase_head=_as_bool(train_raw.get("use_phase_head", env.get("DENSE_TEMPORAL_USE_PHASE_HEAD")), True),
        phase_loss_weight=max(0.0, _as_float(train_raw.get("phase_loss_weight", env.get("DENSE_TEMPORAL_PHASE_LOSS_WEIGHT")), 0.15)),
        ranking_loss_weight=max(0.0, _as_float(train_raw.get("ranking_loss_weight", env.get("DENSE_TEMPORAL_RANKING_LOSS_WEIGHT")), 0.0)),
        cyclece_loss_weight=max(0.0, _as_float(train_raw.get("cyclece_loss_weight", env.get("DENSE_TEMPORAL_CYCLECE_LOSS_WEIGHT")), 1.0)),
        cyclece_tau=max(1e-3, _as_float(train_raw.get("cyclece_tau", env.get("DENSE_TEMPORAL_CYCLECE_TAU")), 0.7)),
        cyclece_radius=max(0, _as_int(train_raw.get("cyclece_radius", env.get("DENSE_TEMPORAL_CYCLECE_RADIUS")), 1)),
        class_loss_weight=max(0.0, _as_float(train_raw.get("class_loss_weight", env.get("DENSE_TEMPORAL_CLASS_LOSS_WEIGHT")), 1.0)),
        class_sampling_alpha=min(1.0, max(0.0, _as_float(train_raw.get("class_sampling_alpha", env.get("DENSE_TEMPORAL_CLASS_SAMPLING_ALPHA")), 0.35))),
        ignore_radius=max(0, _as_int(train_raw.get("ignore_radius", env.get("DENSE_TEMPORAL_IGNORE_RADIUS")), 1)),
        smooth_sigma=max(0.0, _as_float(train_raw.get("smooth_sigma", env.get("DENSE_TEMPORAL_SMOOTH_SIGMA")), 0.0)),
        combine_start_end=_as_bool(train_raw.get("combine_start_end", env.get("DENSE_TEMPORAL_COMBINE_START_END")), False),
        boundary_index_mode=_normalize_boundary_index_mode(
            train_raw.get("boundary_index_mode", env.get("DENSE_TEMPORAL_BOUNDARY_INDEX_MODE") or "nearest")
        ),
        boundary_loss=str(train_raw.get("boundary_loss", env.get("DENSE_TEMPORAL_BOUNDARY_LOSS") or "focal")).strip().lower(),
        gamma=max(0.0, _as_float(train_raw.get("gamma", env.get("DENSE_TEMPORAL_GAMMA")), 2.0)),
        pos_weight_start_end=max(1.0, _as_float(train_raw.get("pos_weight_start_end", env.get("DENSE_TEMPORAL_POS_WEIGHT_SE")), 10.0)),
        pos_weight_cycle=pos_weight_cycle,
        lr=max(1e-7, _as_float(train_raw.get("lr", env.get("DENSE_TEMPORAL_LR")), 1e-3)),
        weight_decay=max(0.0, _as_float(train_raw.get("weight_decay", env.get("DENSE_TEMPORAL_WEIGHT_DECAY")), 1e-4)),
        max_epochs=max(1, _as_int(train_raw.get("max_epochs", env.get("DENSE_TEMPORAL_MAX_EPOCHS")), 120)),
        chunk_len=max(0, _as_int(train_raw.get("chunk_len", env.get("DENSE_TEMPORAL_CHUNK_LEN")), 0)),
        chunks_per_epoch=max(1, _as_int(train_raw.get("chunks_per_epoch", env.get("DENSE_TEMPORAL_CHUNKS_PER_EPOCH")), 64)),
        neg_chunk_fraction=min(1.0, max(0.0, _as_float(train_raw.get("neg_chunk_fraction", env.get("DENSE_TEMPORAL_NEG_CHUNK_FRACTION")), 0.40))),
        grad_clip_norm=max(0.0, _as_float(train_raw.get("grad_clip_norm", env.get("DENSE_TEMPORAL_GRAD_CLIP_NORM")), 1.0)),
        stage1_enabled=_as_bool(train_raw.get("stage1_enabled", env.get("DENSE_TEMPORAL_STAGE1_ENABLED")), True),
        stage1_epochs=max(1, _as_int(train_raw.get("stage1_epochs", env.get("DENSE_TEMPORAL_STAGE1_EPOCHS")), 20)),
        stage1_probe_lr=max(1e-7, _as_float(train_raw.get("stage1_probe_lr", env.get("DENSE_TEMPORAL_STAGE1_PROBE_LR")), 1e-5)),
        stage1_last_block_epoch=max(0, _as_int(train_raw.get("stage1_last_block_epoch", env.get("DENSE_TEMPORAL_STAGE1_LAST_BLOCK_EPOCH")), 8)),
        stage1_all_epoch=max(0, _as_int(train_raw.get("stage1_all_epoch", env.get("DENSE_TEMPORAL_STAGE1_ALL_EPOCH")), 16)),
        stage1_chunks_per_stream=max(1, _as_int(train_raw.get("stage1_chunks_per_stream", env.get("DENSE_TEMPORAL_STAGE1_CHUNKS_PER_STREAM")), 12)),
        stage1_neg_chunk_fraction=min(
            1.0,
            max(
                0.0,
                _as_float(
                    train_raw.get(
                        "stage1_neg_chunk_fraction",
                        env.get("DENSE_TEMPORAL_STAGE1_NEG_CHUNK_FRACTION"),
                    ),
                    0.25,
                ),
            ),
        ),
        stage1_tcn_tune_mode=str(
            train_raw.get("stage1_tcn_tune_mode", env.get("DENSE_TEMPORAL_STAGE1_TCN_TUNE_MODE") or "frozen")
        ).strip().lower(),
        stage1_tcn_last_blocks=max(
            1,
            _as_int(train_raw.get("stage1_tcn_last_blocks", env.get("DENSE_TEMPORAL_STAGE1_TCN_LAST_BLOCKS")), 1),
        ),
        stage1_tcn_lr=max(
            1e-7,
            _as_float(train_raw.get("stage1_tcn_lr", env.get("DENSE_TEMPORAL_STAGE1_TCN_LR")), 5e-5),
        ),
        stage1_tcn_weight_decay=max(
            0.0,
            _as_float(
                train_raw.get("stage1_tcn_weight_decay", env.get("DENSE_TEMPORAL_STAGE1_TCN_WEIGHT_DECAY")),
                1e-4,
            ),
        ),
        stage1_stream_sampling_mode=str(
            train_raw.get("stage1_stream_sampling_mode", env.get("DENSE_TEMPORAL_STAGE1_STREAM_SAMPLING_MODE") or "uniform")
        ).strip().lower(),
        stage1_stream_sampling_power=max(
            0.0,
            _as_float(
                train_raw.get("stage1_stream_sampling_power", env.get("DENSE_TEMPORAL_STAGE1_STREAM_SAMPLING_POWER")),
                0.5,
            ),
        ),
        stage1_stream_sampling_min_weight=max(
            1e-6,
            _as_float(
                train_raw.get(
                    "stage1_stream_sampling_min_weight",
                    env.get("DENSE_TEMPORAL_STAGE1_STREAM_SAMPLING_MIN_WEIGHT"),
                ),
                1.0,
            ),
        ),
        stage1_cyclece_weight=max(0.0, _as_float(train_raw.get("stage1_cyclece_weight", env.get("DENSE_TEMPORAL_STAGE1_CYCLECE_WEIGHT")), 1.0)),
        stage1_smooth_weight=max(0.0, _as_float(train_raw.get("stage1_smooth_weight", env.get("DENSE_TEMPORAL_STAGE1_SMOOTH_WEIGHT")), 0.02)),
        stage1_distill_weight=max(0.0, _as_float(train_raw.get("stage1_distill_weight", env.get("DENSE_TEMPORAL_STAGE1_DISTILL_WEIGHT")), 0.10)),
        stage1_class_weight=max(0.0, _as_float(train_raw.get("stage1_class_weight", env.get("DENSE_TEMPORAL_STAGE1_CLASS_WEIGHT")), 1.0)),
        cyclecl_enabled=_as_bool(train_raw.get("cyclecl_enabled", env.get("DENSE_TEMPORAL_CYCLECL_ENABLED")), False),
        cyclecl_epochs=max(1, _as_int(train_raw.get("cyclecl_epochs", env.get("DENSE_TEMPORAL_CYCLECL_EPOCHS")), 20)),
        cyclecl_patience_epochs=max(0, _as_int(train_raw.get("cyclecl_patience_epochs", env.get("DENSE_TEMPORAL_CYCLECL_PATIENCE_EPOCHS")), 0)),
        cyclecl_max_cycles_per_video=max(
            1,
            _as_int(train_raw.get("cyclecl_max_cycles_per_video", env.get("DENSE_TEMPORAL_CYCLECL_MAX_CYCLES_PER_VIDEO")), 128),
        ),
        cyclecl_cycle_selection_mode=str(
            train_raw.get("cyclecl_cycle_selection_mode", env.get("DENSE_TEMPORAL_CYCLECL_CYCLE_SELECTION_MODE") or "uniform")
        ).strip().lower(),
        cyclecl_min_cycle_frames=max(
            1,
            _as_int(train_raw.get("cyclecl_min_cycle_frames", env.get("DENSE_TEMPORAL_CYCLECL_MIN_CYCLE_FRAMES")), 8),
        ),
        cyclecl_boundary_tolerance_frames=max(
            0,
            _as_int(
                train_raw.get(
                    "cyclecl_boundary_tolerance_frames",
                    env.get("DENSE_TEMPORAL_CYCLECL_BOUNDARY_TOLERANCE_FRAMES"),
                ),
                48,
            ),
        ),
        cyclecl_interior_margin_frames=max(
            0,
            _as_int(
                train_raw.get(
                    "cyclecl_interior_margin_frames",
                    env.get("DENSE_TEMPORAL_CYCLECL_INTERIOR_MARGIN_FRAMES"),
                ),
                12,
            ),
        ),
        cyclecl_mine_batch_size=max(
            1,
            _as_int(train_raw.get("cyclecl_mine_batch_size", env.get("DENSE_TEMPORAL_CYCLECL_MINE_BATCH_SIZE")), 24),
        ),
        cyclecl_triplet_batch_size=max(
            1,
            _as_int(
                train_raw.get("cyclecl_triplet_batch_size", env.get("DENSE_TEMPORAL_CYCLECL_TRIPLET_BATCH_SIZE")),
                12,
            ),
        ),
        cyclecl_max_triplets_per_epoch=max(
            1,
            _as_int(
                train_raw.get(
                    "cyclecl_max_triplets_per_epoch",
                    env.get("DENSE_TEMPORAL_CYCLECL_MAX_TRIPLETS_PER_EPOCH"),
                ),
                4096,
            ),
        ),
        cyclecl_margin=max(0.0, _as_float(train_raw.get("cyclecl_margin", env.get("DENSE_TEMPORAL_CYCLECL_MARGIN")), 0.2)),
        cyclecl_pooler_lr=max(
            1e-7,
            _as_float(train_raw.get("cyclecl_pooler_lr", env.get("DENSE_TEMPORAL_CYCLECL_POOLER_LR")), 1e-4),
        ),
        cyclecl_weight_decay=max(
            0.0,
            _as_float(train_raw.get("cyclecl_weight_decay", env.get("DENSE_TEMPORAL_CYCLECL_WEIGHT_DECAY")), 1e-4),
        ),
        cyclecl_hard_neg_topk=max(
            1,
            _as_int(train_raw.get("cyclecl_hard_neg_topk", env.get("DENSE_TEMPORAL_CYCLECL_HARD_NEG_TOPK")), 8),
        ),
        cyclecl_pos_topk=max(
            1,
            _as_int(train_raw.get("cyclecl_pos_topk", env.get("DENSE_TEMPORAL_CYCLECL_POS_TOPK")), 8),
        ),
        cyclecl_sampling_strategy=str(
            train_raw.get("cyclecl_sampling_strategy", env.get("DENSE_TEMPORAL_CYCLECL_SAMPLING_STRATEGY") or "mean_threshold")
        ).strip().lower(),
        cyclecl_mean_threshold_beta=max(
            0.0,
            _as_float(
                train_raw.get(
                    "cyclecl_mean_threshold_beta",
                    env.get("DENSE_TEMPORAL_CYCLECL_MEAN_THRESHOLD_BETA"),
                ),
                0.3,
            ),
        ),
        cyclecl_grad_clip_norm=max(
            0.0,
            _as_float(train_raw.get("cyclecl_grad_clip_norm", env.get("DENSE_TEMPORAL_CYCLECL_GRAD_CLIP_NORM")), 1.0),
        ),
        stage1_fail_if_best_epoch_zero=_as_bool(
            train_raw.get("stage1_fail_if_best_epoch_zero", env.get("DENSE_TEMPORAL_STAGE1_FAIL_IF_EPOCH0")),
            False,
        ),
        seed=_as_int(train_raw.get("seed", env.get("DENSE_TEMPORAL_SEED")), 42),
    )

    return DenseTemporalRunConfig(
        run_id=str(run_id),
        space_id=str(space_id),
        run_number=int(run_number),
        dataset_mode=dataset_mode,
        temporal_structure_mode=temporal_structure_mode,
        videos=videos,
        temporal_targets=temporal_targets,
        model=model,
        train=train,
    )
