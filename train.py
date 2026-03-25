"""
Dual-space autoresearch dense-temporal trainer for the Minda TCN spaces.

This run keeps the production dense-temporal logic intact as much as possible:

- one boundary TCN per space
- one probe phase-1 finetune per space
- production target preparation, losses, checkpoint formats, and evaluation

The only intentional model change versus production is:

- `tcn_bidirectional = False`

The two spaces train as independent models, but they advance in lockstep inside
the fixed 10-minute autoresearch budget:

- 5 minutes total for TCN rounds
- 5 minutes total for probe rounds

The prepared cache owns the 50/50 camera/video-balanced split. This trainer
consumes the cached split files and does not rebuild train/validation partitions
at runtime.
"""

from __future__ import annotations

import copy
import importlib.util
import importlib.machinery
import io
import json
import logging
import os
import random
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.cache_contract import (
    CACHE_VERSION,
    DEFAULT_SPLIT_POLICY,
    TIME_BUDGET,
    TOTAL_TIMEOUT_SECONDS,
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
from autoresearch_vjepa.contracts import DenseTemporalRunConfig, parse_dense_temporal_snapshot


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

from src.training.dense_temporal import probe_phase1 as prod_probe
from src.training.dense_temporal import tcn_train as prod_tcn

LOGGER = logging.getLogger("autoresearch.minda.train")

MODEL_FAMILY = "minda_dual_prod_tcn_probe_phase1"
SEED = 42

SPACE_ORDER = ("minda-button-tcn", "minda-subassembly-tcn")
SPACE_KEY_BY_NAME = {
    "minda-button-tcn": "button",
    "minda-subassembly-tcn": "subassembly",
}

DEFAULT_BUTTON_SOURCE_RUN_DIR = Path(
    "/tmp/embedding_runs/a8c6f52b-cb31-4942-b1df-5b30f3ed5b81-hybrid-rnd-20260324T124730Z/dense_temporal"
)
DEFAULT_SUBASSEMBLY_SOURCE_RUN_DIR = Path(
    "/tmp/embedding_runs/78768e0e-e985-4eca-9f26-8757b7f7209a/dense_temporal"
)
DEFAULT_SUBASSEMBLY_MATERIALIZED_DIR = Path(
    "/tmp/autoresearch_minda_sources/minda-subassembly-tcn/dense_temporal"
)

DEFAULT_BUTTON_CACHE_DIR = Path("/tmp/autoresearch-minda-button-cache")
DEFAULT_SUBASSEMBLY_CACHE_DIR = Path("/tmp/autoresearch-minda-subassembly-cache")
DEFAULT_OUTPUT_ROOT = Path("/tmp/autoresearch_minda_runs")

DEFAULT_TCN_STAGE_SECONDS = 300.0
DEFAULT_PROBE_STAGE_SECONDS = 300.0
DEFAULT_PROBE_EVAL_TOKEN_CHUNK = 32


def _resolve_vjepa_vendor_root() -> Path:
    vendor_root = (WORKSPACE_ROOT / "third_party" / "vjepa2_testing").resolve()
    if not vendor_root.exists():
        raise FileNotFoundError(f"Missing vendored V-JEPA root: {vendor_root}")
    return vendor_root


@contextmanager
def _use_vjepa_vendor_src_namespace():
    """Temporarily expose the vendored V-JEPA namespace package layout.

    The vendor code expects top-level imports from a namespace package named
    `src`, but this autoresearch entrypoint already imported the main repo's
    regular `src` package for production training modules. Swap the import view
    only while the probe pooler is being constructed, then restore the main repo
    modules immediately afterwards.
    """

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
    module_name = f"_minda_vjepa_model_utils_{time.time_ns()}"
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
    key: str
    source_run_dir: Path
    cache_root: Path
    output_dir: Path
    cfg: DenseTemporalRunConfig
    initial_pooler_path: Path
    encoder_checkpoint: str
    encoder_model: str
    split_plan: SplitPlan


@dataclass
class EvalBundle:
    metrics_by_space: Dict[str, Dict[str, float]]
    primary_metric: float
    mean_pair_f1: float
    mean_count_mae: float
    mean_start_mae_ms: float
    mean_end_mae_ms: float


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
    probe_raw = os.getenv("AUTORESEARCH_PROBE_STAGE_SECONDS", "").strip()
    tcn_seconds = float(tcn_raw) if tcn_raw else float(DEFAULT_TCN_STAGE_SECONDS)
    probe_seconds = float(probe_raw) if probe_raw else float(DEFAULT_PROBE_STAGE_SECONDS)
    requested = max(1e-6, float(tcn_seconds + probe_seconds))
    scale = float(total_budget_seconds) / float(requested)
    return float(tcn_seconds * scale), float(probe_seconds * scale)


def _normalize_workspace_path(raw: str) -> Path:
    text = str(raw or "").strip()
    if text.startswith("/workspace/"):
        return (WORKSPACE_ROOT / text[len("/workspace/") :]).resolve()
    return Path(text).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _merge_effective_snapshot(snapshot_path: Path, resolved_config_path: Path) -> Dict[str, Any]:
    snapshot = _load_json(snapshot_path)
    if isinstance(snapshot.get("temporal_model"), dict) and isinstance(snapshot.get("temporal_training"), dict):
        return snapshot
    resolved = _load_json(resolved_config_path)
    merged = dict(snapshot)
    fallback_fields = {
        "dataset_mode": resolved.get("dataset_mode"),
        "temporal_structure_mode": resolved.get("temporal_structure_mode"),
        "temporal_targets": resolved.get("temporal_targets"),
        "temporal_model": resolved.get("model"),
        "temporal_training": resolved.get("train"),
        "videos": resolved.get("videos"),
    }
    for key, value in fallback_fields.items():
        current = merged.get(key)
        if current in (None, {}, [], "") and value is not None:
            merged[key] = value
    return merged


def _resolve_source_candidates(space_name: str) -> List[Path]:
    key = SPACE_KEY_BY_NAME[space_name]
    env_key = f"AUTORESEARCH_MINDA_{key.upper()}_SOURCE_RUN_DIR"
    override = os.getenv(env_key, "").strip()
    if override:
        return [Path(override).expanduser()]
    if space_name == "minda-button-tcn":
        return [DEFAULT_BUTTON_SOURCE_RUN_DIR]
    return [DEFAULT_SUBASSEMBLY_MATERIALIZED_DIR, DEFAULT_SUBASSEMBLY_SOURCE_RUN_DIR]


def _source_materialization_complete(dense_root: Path) -> Tuple[bool, Optional[str]]:
    status_path = dense_root / "materialization_status.json"
    if not status_path.exists():
        return True, None
    try:
        payload = _load_json(status_path)
    except Exception as exc:
        return False, f"Unreadable materialization status at {status_path}: {exc}"
    if bool(payload.get("complete")):
        return True, None
    stop_reason = str(payload.get("stop_reason") or "unknown")
    completed = int(payload.get("jobs_completed") or 0)
    total = int(payload.get("jobs_total") or 0)
    return False, f"incomplete source ({completed}/{total} videos, stop_reason={stop_reason})"


def _resolve_source_run_dir(space_name: str) -> Path:
    allow_partial = os.getenv("AUTORESEARCH_ALLOW_PARTIAL_SOURCES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    for candidate in _resolve_source_candidates(space_name):
        dense_root = candidate.expanduser().resolve()
        if (dense_root / "features").exists() and (dense_root / "labels").exists():
            complete, reason = _source_materialization_complete(dense_root)
            if not complete and not allow_partial:
                raise RuntimeError(
                    f"Prepared source root for {space_name} is present but {reason}. "
                    "Set AUTORESEARCH_ALLOW_PARTIAL_SOURCES=1 only for smoke tests."
                )
            return dense_root
    candidates = ", ".join(str(path) for path in _resolve_source_candidates(space_name))
    raise FileNotFoundError(
        f"No prepared dense_temporal source root found for {space_name}. "
        f"Checked: {candidates}"
    )


def _resolve_cache_root(space_name: str) -> Path:
    key = SPACE_KEY_BY_NAME[space_name]
    env_key = f"AUTORESEARCH_MINDA_{key.upper()}_CACHE_DIR"
    override = os.getenv(env_key, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (
        DEFAULT_BUTTON_CACHE_DIR if space_name == "minda-button-tcn" else DEFAULT_SUBASSEMBLY_CACHE_DIR
    ).resolve()


def _resolve_output_root() -> Path:
    override = os.getenv("AUTORESEARCH_OUTPUT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def _resolve_initial_pooler_path(source_run_dir: Path, cfg: DenseTemporalRunConfig) -> Path:
    staged = source_run_dir / "staged" / "pooler_pretrained.pt"
    if staged.exists():
        return staged.resolve()
    return _normalize_workspace_path(str(cfg.model.pooler_path))


def _patch_cfg_for_experiment(cfg: DenseTemporalRunConfig) -> DenseTemporalRunConfig:
    return replace(
        cfg,
        model=replace(
            cfg.model,
            pooler_path=str(_normalize_workspace_path(str(cfg.model.pooler_path))),
            encoder_checkpoint=str(_normalize_workspace_path(str(cfg.model.encoder_checkpoint))),
        ),
        train=replace(cfg.train, tcn_bidirectional=False),
    )


def _load_space_cfg(source_run_dir: Path) -> DenseTemporalRunConfig:
    resolved = _load_json(source_run_dir / "resolved_config.json")
    snapshot = _merge_effective_snapshot(
        source_run_dir / "snapshot.json",
        source_run_dir / "resolved_config.json",
    )
    cfg = parse_dense_temporal_snapshot(
        run_id=str(resolved.get("run_id") or source_run_dir.parent.name),
        space_id=str(resolved.get("space_id") or ""),
        run_number=int(resolved.get("run_number") or 0),
        snapshot=snapshot,
        env=dict(os.environ),
    )
    return _patch_cfg_for_experiment(cfg)


def _ensure_cache(space_name: str, source_run_dir: Path, cache_root_base: Path) -> Path:
    cache_root = cache_root_base / CACHE_VERSION
    manifest_path = cache_root / "manifest.json"
    need_build = not manifest_path.exists()
    if not need_build:
        try:
            manifest = load_manifest(cache_root=cache_root)
            sources = {str(Path(item).resolve()) for item in manifest.get("source_run_dirs", [])}
            manifest_policy = str(manifest.get("split_policy") or "").strip()
            need_build = (
                str(source_run_dir.resolve()) not in sources
                or manifest_policy != DEFAULT_SPLIT_POLICY
            )
        except Exception:
            need_build = True
    if need_build:
        LOGGER.info("Building autoresearch cache for %s from %s", space_name, source_run_dir)
        configure_cache_paths(cache_root)
        build_cache(
            source_run_dirs=[str(source_run_dir)],
            source_globs=[],
            camera_include_regex=None,
            video_include_regex=None,
            path_include_regex=None,
            split_policy=DEFAULT_SPLIT_POLICY,
            val_ratio=0.5,
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


def _load_cached_split_plan(cache_root: Path) -> SplitPlan:
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
        target_val_videos=(
            float(manifest["split_target_val_videos"])
            if manifest.get("split_target_val_videos") is not None
            else float(len(train_videos) + len(val_videos)) / 2.0
        ),
    )


def _build_space_spec(space_name: str, output_root: Path) -> SpaceSpec:
    source_run_dir = _resolve_source_run_dir(space_name)
    cfg = _load_space_cfg(source_run_dir)
    cache_root = _ensure_cache(space_name, source_run_dir, _resolve_cache_root(space_name))
    split_plan = _load_cached_split_plan(cache_root)
    output_dir = output_root / SPACE_KEY_BY_NAME[space_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    return SpaceSpec(
        name=space_name,
        key=SPACE_KEY_BY_NAME[space_name],
        source_run_dir=source_run_dir,
        cache_root=cache_root,
        output_dir=output_dir,
        cfg=cfg,
        initial_pooler_path=_resolve_initial_pooler_path(source_run_dir, cfg),
        encoder_checkpoint=str(_normalize_workspace_path(str(cfg.model.encoder_checkpoint))),
        encoder_model=str(cfg.model.encoder_model),
        split_plan=split_plan,
    )


def _clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _count_trainable_params(module: nn.Module) -> int:
    return int(sum(int(param.numel()) for param in module.parameters() if param.requires_grad))


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _score_eval_bundle(bundle: EvalBundle) -> Tuple[float, float, float, float, float]:
    return (
        float(bundle.primary_metric),
        float(bundle.mean_pair_f1),
        -float(bundle.mean_count_mae),
        -float(bundle.mean_start_mae_ms),
        -float(bundle.mean_end_mae_ms),
    )


def _should_replace_eval(best: Optional[EvalBundle], candidate: EvalBundle) -> bool:
    if best is None:
        return True
    return _score_eval_bundle(candidate) > _score_eval_bundle(best)


def _bundle_metrics(metrics_by_space: Mapping[str, Mapping[str, float]]) -> EvalBundle:
    pair_f1s = [float(metrics["val_pair_f1"]) for metrics in metrics_by_space.values()]
    count_maes = [float(metrics["val_count_mae"]) for metrics in metrics_by_space.values()]
    start_maes = [float(metrics["val_start_mae_ms"]) for metrics in metrics_by_space.values()]
    end_maes = [float(metrics["val_end_mae_ms"]) for metrics in metrics_by_space.values()]
    return EvalBundle(
        metrics_by_space={key: dict(value) for key, value in metrics_by_space.items()},
        primary_metric=float(min(pair_f1s)),
        mean_pair_f1=float(np.mean(pair_f1s)),
        mean_count_mae=float(np.mean(count_maes)),
        mean_start_mae_ms=float(np.mean(start_maes)),
        mean_end_mae_ms=float(np.mean(end_maes)),
    )


def _tcn_heads(base_heads: int, combine: bool) -> Tuple[str, ...]:
    if combine:
        return ("boundary", "cycle")[:base_heads]
    return ("start", "end", "cycle")[:base_heads]


@dataclass
class _TCNPreparedState:
    streams: List[Any]
    stream_class_targets: List[np.ndarray]
    stream_class_masks: List[np.ndarray]
    class_index_to_label_id: List[int]
    class_index_to_label_name: List[str]
    global_action_label_ids: List[int]
    global_action_label_name_by_id: Dict[int, str]
    num_class_logits: int
    class_offset: int
    phase_offset: int
    class_weights_np: np.ndarray
    class_weights_t: Optional[torch.Tensor]
    pos_weight_cycle: float
    pos_weight_start_end: float
    base_heads: int
    out_dim: int
    boundary_loss_name: str
    use_phase_head: bool
    combine: bool
    model_cfg_dict: Dict[str, Any]
    left_ctx: int


class TCNTrainer:
    def __init__(self, spec: SpaceSpec, *, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self.cfg = prod_tcn.BoundaryTrainConfig(
            seq_model=str(spec.cfg.train.seq_model),
            hidden_dim=int(spec.cfg.train.hidden_dim),
            kernel_size=int(spec.cfg.train.kernel_size),
            dropout=float(spec.cfg.train.dropout),
            use_layernorm=bool(spec.cfg.train.use_layernorm),
            dilations=tuple(int(item) for item in spec.cfg.train.dilations),
            bidirectional=False,
            task_specific_heads=bool(spec.cfg.train.tcn_task_specific_heads),
            boundary_loss=str(spec.cfg.train.boundary_loss),
            gamma=float(spec.cfg.train.gamma),
            pos_weight_start_end=float(spec.cfg.train.pos_weight_start_end),
            pos_weight_cycle=spec.cfg.train.pos_weight_cycle,
            use_phase_head=bool(spec.cfg.train.use_phase_head),
            phase_loss_weight=float(spec.cfg.train.phase_loss_weight),
            ranking_loss_weight=float(spec.cfg.train.ranking_loss_weight),
            cyclece_loss_weight=float(spec.cfg.train.cyclece_loss_weight),
            cyclece_tau=float(spec.cfg.train.cyclece_tau),
            cyclece_radius=int(spec.cfg.train.cyclece_radius),
            class_loss_weight=float(spec.cfg.train.class_loss_weight),
            class_sampling_alpha=float(spec.cfg.train.class_sampling_alpha),
            temporal_structure_mode=str(spec.cfg.temporal_structure_mode),
            ignore_radius=int(spec.cfg.train.ignore_radius),
            smooth_sigma=float(spec.cfg.train.smooth_sigma),
            combine_start_end=bool(spec.cfg.train.combine_start_end),
            boundary_index_mode=str(spec.cfg.train.boundary_index_mode),
            lr=float(spec.cfg.train.lr),
            weight_decay=float(spec.cfg.train.weight_decay),
            max_epochs=int(spec.cfg.train.max_epochs),
            seed=int(spec.cfg.train.seed),
            device=str(device),
            grad_clip_norm=float(spec.cfg.train.grad_clip_norm),
            chunk_len=0,
            chunks_per_epoch=int(spec.cfg.train.chunks_per_epoch),
            neg_chunk_fraction=float(spec.cfg.train.neg_chunk_fraction),
        )
        self.state = self._prepare_state()
        self.model, model_cfg_dict, left_ctx = prod_tcn._build_model_and_cfg(
            cfg=self.cfg,
            input_dim=int(self.state.streams[0].embeddings.shape[1]),
            out_dim=int(self.state.out_dim),
            device=self.device,
        )
        self.state.model_cfg_dict = dict(model_cfg_dict)
        self.state.left_ctx = int(left_ctx)
        self.optimizer = self._build_optimizer()
        self.pos_w_se_t = torch.tensor(float(self.state.pos_weight_start_end), device=self.device)
        self.pos_w_cycle_t = torch.tensor(float(self.state.pos_weight_cycle), device=self.device)
        self.history: List[Dict[str, float]] = []
        self.epoch = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        if self.device.type == "cuda":
            try:
                return torch.optim.AdamW(
                    self.model.parameters(),
                    lr=float(self.cfg.lr),
                    weight_decay=float(self.cfg.weight_decay),
                    fused=True,
                )
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _prepare_state(self) -> _TCNPreparedState:
        streams = [
            prod_tcn._prepare_supervised_stream(
                name=str(record.segment_id),
                embeddings_path=Path(record.feature_path),
                labels_path=Path(record.label_path),
                cfg=self.cfg,
            )
            for record in self.spec.split_plan.train_records
        ]
        input_dim = int(streams[0].embeddings.shape[1])
        for stream in streams[1:]:
            if int(stream.embeddings.shape[1]) != input_dim:
                raise ValueError("Embedding dim mismatch across streams.")

        y_cycle_all = np.concatenate([stream.y_cycle[stream.mask_cycle > 0.5] for stream in streams], axis=0)
        pos_weight_cycle = float(prod_tcn._resolve_cycle_pos_weight(y_cycle_all, self.cfg))
        combine = bool(self.cfg.combine_start_end)
        base_heads = 2 if combine else 3
        use_phase_head = bool(self.cfg.use_phase_head)

        idle_label_counts: Dict[int, int] = {}
        global_action_label_ids: List[int] = []
        global_action_label_name_by_id: Dict[int, str] = {}
        for stream in streams:
            idle_id = int(stream.fallback_idle_label_id)
            idle_label_counts[idle_id] = idle_label_counts.get(idle_id, 0) + 1
            for label_id in stream.action_label_ids:
                lid = int(label_id)
                if lid in {int(stream.ignore_label_id), int(stream.fallback_idle_label_id)}:
                    continue
                if lid not in global_action_label_ids:
                    global_action_label_ids.append(lid)
                label_name = prod_tcn._normalize_action_label_name(stream.action_label_name_by_id.get(lid))
                if label_name and lid not in global_action_label_name_by_id:
                    global_action_label_name_by_id[lid] = str(label_name)
        if not global_action_label_ids:
            global_action_label_ids = [1]

        global_idle_label_id = (
            max(idle_label_counts.items(), key=lambda item: item[1])[0] if idle_label_counts else 0
        )
        global_action_id_to_class_index = {
            int(label_id): int(index + 1) for index, label_id in enumerate(global_action_label_ids)
        }
        num_class_logits = int(1 + len(global_action_label_ids))
        class_offset = int(base_heads)
        phase_offset = int(base_heads + num_class_logits)
        class_index_to_label_id = [int(global_idle_label_id)] + [int(item) for item in global_action_label_ids]
        class_index_to_label_name = ["idle"] + [
            str(global_action_label_name_by_id.get(int(item), "")) for item in global_action_label_ids
        ]

        stream_class_targets: List[np.ndarray] = []
        stream_class_masks: List[np.ndarray] = []
        for stream in streams:
            raw_ids = stream.y_class_label_id.astype(np.int64, copy=False)
            class_idx = np.zeros(raw_ids.shape, dtype=np.int64)
            for action_label_id, class_idx_val in global_action_id_to_class_index.items():
                class_idx[raw_ids == int(action_label_id)] = int(class_idx_val)
            mask_cls = stream.mask_class.astype(np.float32, copy=False).copy()
            mask_cls[raw_ids == int(stream.ignore_label_id)] = 0.0
            stream_class_targets.append(class_idx)
            stream_class_masks.append(mask_cls)

        supervised_cls = np.concatenate(
            [
                stream_class_targets[idx][stream_class_masks[idx] > 0.5]
                for idx in range(len(streams))
                if stream_class_masks[idx].size > 0
            ],
            axis=0,
        )
        class_weights_np = np.ones((num_class_logits,), dtype=np.float32)
        if supervised_cls.size > 0:
            counts = np.bincount(
                supervised_cls.astype(np.int64, copy=False),
                minlength=num_class_logits,
            ).astype(np.float32)
            inv = np.zeros_like(counts)
            inv[counts > 0] = 1.0 / np.sqrt(counts[counts > 0])
            if float(np.mean(inv[counts > 0])) > 0.0:
                inv = inv / float(np.mean(inv[counts > 0]))
            inv[inv <= 0.0] = 1.0
            class_weights_np = inv.astype(np.float32, copy=False)
        class_weights_t = (
            torch.from_numpy(class_weights_np).to(self.device)
            if float(self.cfg.class_loss_weight) > 0.0 and num_class_logits > 1
            else None
        )

        out_dim = int(base_heads + num_class_logits + (2 if use_phase_head else 0))
        return _TCNPreparedState(
            streams=streams,
            stream_class_targets=stream_class_targets,
            stream_class_masks=stream_class_masks,
            class_index_to_label_id=class_index_to_label_id,
            class_index_to_label_name=class_index_to_label_name,
            global_action_label_ids=global_action_label_ids,
            global_action_label_name_by_id=global_action_label_name_by_id,
            num_class_logits=num_class_logits,
            class_offset=class_offset,
            phase_offset=phase_offset,
            class_weights_np=class_weights_np,
            class_weights_t=class_weights_t,
            pos_weight_cycle=pos_weight_cycle,
            pos_weight_start_end=float(self.cfg.pos_weight_start_end),
            base_heads=base_heads,
            out_dim=out_dim,
            boundary_loss_name=prod_tcn._normalize_boundary_loss(self.cfg.boundary_loss),
            use_phase_head=use_phase_head,
            combine=combine,
            model_cfg_dict={},
            left_ctx=0,
        )

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        combine = bool(self.state.combine)
        use_phase_head = bool(self.state.use_phase_head)
        class_weight_lambda = float(max(0.0, self.cfg.class_loss_weight))
        phase_weight = float(self.cfg.phase_loss_weight)
        cyclece_weight = float(self.cfg.cyclece_loss_weight)

        conf_epoch = np.zeros((self.state.num_class_logits, self.state.num_class_logits), dtype=np.int64)
        total_mask = torch.tensor(0.0, device=self.device)
        total_cycle_T = torch.tensor(0.0, device=self.device)
        total_class_T = torch.tensor(0.0, device=self.device)
        total_phase_T = torch.tensor(0.0, device=self.device)
        cycle_sum = torch.tensor(0.0, device=self.device)
        class_sum = torch.tensor(0.0, device=self.device)
        phase_sum = torch.tensor(0.0, device=self.device)
        cyclece_sum = torch.tensor(0.0, device=self.device)
        cyclece_count = torch.tensor(0.0, device=self.device)
        boundary_sum = torch.tensor(0.0, device=self.device)
        start_sum = torch.tensor(0.0, device=self.device)
        end_sum = torch.tensor(0.0, device=self.device)
        total_T = torch.tensor(0.0, device=self.device)
        p_mean_sum = torch.zeros(self.state.base_heads, device=self.device)
        p_max = torch.zeros(self.state.base_heads, device=self.device)

        for idx, stream in enumerate(self.state.streams):
            x = _torch_from_numpy_safe(stream.embeddings, device=self.device).unsqueeze(0)
            mask = torch.from_numpy(stream.mask_start_end).unsqueeze(0).to(self.device)
            mask_cycle = torch.from_numpy(stream.mask_cycle).unsqueeze(0).to(self.device)
            mask_class = torch.from_numpy(self.state.stream_class_masks[idx]).unsqueeze(0).to(self.device)
            y_cycle = torch.from_numpy(stream.y_cycle).unsqueeze(0).to(self.device)
            y_class = (
                torch.from_numpy(self.state.stream_class_targets[idx]).unsqueeze(0).to(self.device).long()
            )
            logits = self.model(x)

            mask_count = mask.sum()
            cycle_count = mask_cycle.sum()
            class_count = mask_class.sum()
            phase_mask = torch.from_numpy(stream.mask_phase).unsqueeze(0).to(self.device)
            phase_count = phase_mask.sum()
            t_count = torch.tensor(float(stream.embeddings.shape[0]), device=self.device)

            if combine:
                y_boundary = torch.from_numpy(stream.y_boundary).unsqueeze(0).to(self.device)
                logits_boundary = logits[:, :, 0]
                logits_cycle = logits[:, :, 1]
                loss_boundary_raw = prod_tcn._boundary_loss_with_logits(
                    logits=logits_boundary,
                    targets=y_boundary,
                    pos_weight=self.pos_w_se_t,
                    gamma=float(self.cfg.gamma),
                    loss_name=str(self.state.boundary_loss_name),
                )
                loss_boundary = prod_tcn.masked_mean(loss_boundary_raw, mask)
                loss_cycle_raw = prod_tcn.bce_with_logits(
                    logits_cycle,
                    y_cycle,
                    pos_weight=self.pos_w_cycle_t if float(self.state.pos_weight_cycle) != 1.0 else None,
                    reduction="none",
                )
                loss_cycle = prod_tcn.masked_mean(loss_cycle_raw, mask_cycle)
                boundary_sum = boundary_sum + (loss_boundary * mask_count)
            else:
                y_se = torch.from_numpy(np.stack([stream.y_start, stream.y_end], axis=-1)).unsqueeze(0).to(self.device)
                logits_start_end = logits[:, :, 0:2]
                logits_cycle = logits[:, :, 2]
                loss_se_raw = prod_tcn._boundary_loss_with_logits(
                    logits=logits_start_end,
                    targets=y_se,
                    pos_weight=self.pos_w_se_t,
                    gamma=float(self.cfg.gamma),
                    loss_name=str(self.state.boundary_loss_name),
                )
                loss_start = prod_tcn.masked_mean(loss_se_raw[:, :, 0], mask)
                loss_end = prod_tcn.masked_mean(loss_se_raw[:, :, 1], mask)
                loss_cycle_raw = prod_tcn.bce_with_logits(
                    logits_cycle,
                    y_cycle,
                    pos_weight=self.pos_w_cycle_t if float(self.state.pos_weight_cycle) != 1.0 else None,
                    reduction="none",
                )
                loss_cycle = prod_tcn.masked_mean(loss_cycle_raw, mask_cycle)
                start_sum = start_sum + (loss_start * mask_count)
                end_sum = end_sum + (loss_end * mask_count)

            logits_class = logits[
                :, :, self.state.class_offset : self.state.class_offset + self.state.num_class_logits
            ]
            loss_class = prod_tcn._masked_multiclass_ce(
                logits=logits_class,
                targets=y_class,
                mask=mask_class,
                class_weights=self.state.class_weights_t,
            )
            class_sum = class_sum + (loss_class * class_count)

            loss_phase = logits.new_tensor(0.0)
            if use_phase_head and phase_weight > 0.0:
                y_phase = torch.from_numpy(stream.y_phase).unsqueeze(0).to(self.device)
                logits_phase = logits[:, :, self.state.phase_offset : self.state.phase_offset + 2]
                loss_phase = prod_tcn._phase_loss_masked(
                    pred_phase_logits=logits_phase,
                    target_phase=y_phase,
                    mask_phase=phase_mask,
                    loss_name=str(self.cfg.phase_loss),
                    huber_delta=float(self.cfg.phase_huber_delta),
                )
                phase_sum = phase_sum + (loss_phase * phase_count)

            loss_cyclece = logits.new_tensor(0.0)
            if cyclece_weight > 0.0 and stream.mapped_cycles_idx:
                loss_cyclece = prod_tcn._cycle_ce_loss(
                    logits=logits.squeeze(0)[:, : self.state.base_heads],
                    mapped_cycles_idx=stream.mapped_cycles_idx,
                    combine=bool(combine),
                    tau=float(self.cfg.cyclece_tau),
                    radius=int(self.cfg.cyclece_radius),
                )
                cyclece_sum = cyclece_sum + loss_cyclece
                cyclece_count = cyclece_count + torch.tensor(1.0, device=self.device)

            total_mask = total_mask + mask_count
            total_cycle_T = total_cycle_T + cycle_count
            total_class_T = total_class_T + class_count
            total_phase_T = total_phase_T + phase_count
            cycle_sum = cycle_sum + (loss_cycle * cycle_count)
            total_T = total_T + t_count

            with torch.no_grad():
                boundary_probs = torch.sigmoid(logits[:, :, : self.state.base_heads]).squeeze(0)
                p_mean_sum = p_mean_sum + (boundary_probs.mean(dim=0) * t_count)
                p_max = torch.maximum(p_max, boundary_probs.max(dim=0).values)
                pred_cls = torch.argmax(logits_class.squeeze(0), dim=-1).detach().cpu().numpy()
                conf_delta, _, _ = prod_tcn._compute_multiclass_metrics(
                    y_true=self.state.stream_class_targets[idx],
                    y_pred=pred_cls,
                    mask=self.state.stream_class_masks[idx],
                    num_classes=self.state.num_class_logits,
                )
                conf_epoch += conf_delta

        loss_cycle_g = cycle_sum / torch.clamp(total_cycle_T, min=1.0)
        loss_class_g = class_sum / torch.clamp(total_class_T, min=1.0)
        loss_phase_g = (
            phase_sum / torch.clamp(total_phase_T, min=1.0)
            if use_phase_head and phase_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )
        loss_cyclece_g = (
            cyclece_sum / torch.clamp(cyclece_count, min=1.0)
            if cyclece_weight > 0.0
            else torch.tensor(0.0, device=self.device)
        )
        if combine:
            loss_boundary_g = boundary_sum / torch.clamp(total_mask, min=1.0)
            total_loss = (
                loss_boundary_g
                + loss_cycle_g
                + (loss_class_g * class_weight_lambda)
                + (loss_phase_g * phase_weight)
                + (loss_cyclece_g * cyclece_weight)
            )
        else:
            loss_start_g = start_sum / torch.clamp(total_mask, min=1.0)
            loss_end_g = end_sum / torch.clamp(total_mask, min=1.0)
            total_loss = (
                loss_start_g
                + loss_end_g
                + loss_cycle_g
                + (loss_class_g * class_weight_lambda)
                + (loss_phase_g * phase_weight)
                + (loss_cyclece_g * cyclece_weight)
            )

        total_loss.backward()
        if float(self.cfg.grad_clip_norm or 0.0) > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg.grad_clip_norm))
        self.optimizer.step()

        per_class, macro_f1 = prod_tcn._per_class_metrics_from_confusion(conf_epoch)
        row: Dict[str, float] = {
            "epoch": float(self.epoch),
            "loss": float(total_loss.item()),
            "loss_cycle": float(loss_cycle_g.item()),
            "loss_class": float(loss_class_g.item()),
            "loss_phase": float(loss_phase_g.item()),
            "loss_cyclece": float(loss_cyclece_g.item()),
            "class_macro_f1": float(macro_f1),
        }
        if combine:
            row["loss_boundary"] = float(loss_boundary_g.item())
        else:
            row["loss_start"] = float(loss_start_g.item())
            row["loss_end"] = float(loss_end_g.item())
        p_means = (p_mean_sum / torch.clamp(total_T, min=1.0)).detach().cpu().numpy().tolist()
        p_max_vals = p_max.detach().cpu().numpy().tolist()
        for idx, head_name in enumerate(_tcn_heads(self.state.base_heads, self.state.combine)):
            row[f"p_mean_{head_name}"] = float(p_means[idx])
            row[f"p_max_{head_name}"] = float(p_max_vals[idx])
        self.history.append(row)
        self.epoch += 1
        LOGGER.info(
            "TCN space=%s epoch=%d loss=%.4f pair_heads=%s macro_f1=%.4f trainable_params_M=%.3f",
            self.spec.name,
            self.epoch,
            row["loss"],
            list(_tcn_heads(self.state.base_heads, self.state.combine)),
            row["class_macro_f1"],
            _count_trainable_params(self.model) / 1e6,
        )
        return row

    def _predict_pairs_for_record(self, record: SegmentRecord) -> List[EventPair]:
        payload = load_segment_arrays(record, representation="pooled_z0", use_eval_span=True)
        logits = prod_tcn._predict_full(
            self.model,
            payload["pooled_z0"],
            left_ctx=int(self.state.left_ctx),
            chunk_len=0,
            device=self.device,
        )
        probs = 1.0 / (1.0 + np.exp(-logits[:, : self.state.base_heads]))
        return decode_event_pairs(
            probs,
            payload["timestamps_ms"],
            heads=_tcn_heads(self.state.base_heads, self.state.combine),
        )

    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        predictions = {
            record.segment_id: self._predict_pairs_for_record(record)
            for record in self.spec.split_plan.val_eval_records
        }
        return evaluate_predictions(predictions, records=self.spec.split_plan.val_eval_records)

    def build_checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "seq_model": prod_tcn._normalize_seq_model(str(self.cfg.seq_model)),
            "model_state": {key: value.detach().cpu() for key, value in self.model.state_dict().items()},
            "model_cfg": dict(self.state.model_cfg_dict),
            "train_cfg": asdict(self.cfg),
            "temporal_structure_mode": str(self.spec.cfg.temporal_structure_mode),
            "effective_losses": {
                "boundary": True,
                "class": float(self.cfg.class_loss_weight) > 0.0,
                "cycle": True,
                "phase": bool(self.cfg.use_phase_head) and float(self.cfg.phase_loss_weight) > 0.0,
                "ranking": False,
                "cyclece": float(self.cfg.cyclece_loss_weight) > 0.0,
            },
            "heads": prod_tcn._build_output_heads(
                combine=bool(self.state.combine),
                use_phase_head=bool(self.state.use_phase_head),
                class_index_to_label_id=self.state.class_index_to_label_id,
                class_index_to_label_name=self.state.class_index_to_label_name,
            ),
            "streams": [
                {
                    "name": stream.name,
                    "embeddings_path": str(stream.embeddings_path),
                    "labels_path": str(stream.labels_path),
                    "supervised_mode": str(stream.supervised_mode),
                    "mapping_stats": stream.mapping_stats,
                    "cycle_pos_frac": float(stream.cycle_pos_frac),
                    "cycle_supervised_frac": float(stream.cycle_supervised_frac),
                    "valid_start_idx_full": int(stream.valid_start_idx_full),
                    "valid_end_idx_full": int(stream.valid_end_idx_full),
                    "full_len": int(stream.full_len),
                }
                for stream in self.state.streams
            ],
            "multiclass_head": {
                "enabled": float(self.cfg.class_loss_weight) > 0.0,
                "class_offset": int(self.state.class_offset),
                "num_classes": int(self.state.num_class_logits),
                "class_index_to_label_id": [int(item) for item in self.state.class_index_to_label_id],
                "class_index_to_label_name": [str(item) for item in self.state.class_index_to_label_name],
                "label_id_to_label_name": {
                    str(label_id): str(label_name)
                    for label_id, label_name in self.state.global_action_label_name_by_id.items()
                    if str(label_name).strip()
                },
                "action_label_ids": [int(item) for item in self.state.global_action_label_ids],
                "loss_weight": float(self.cfg.class_loss_weight),
                "class_weights": [float(item) for item in self.state.class_weights_np.tolist()],
            },
            "pos_weight_cycle": float(self.state.pos_weight_cycle),
            "pos_weight_start_end": float(self.state.pos_weight_start_end),
        }

    def save_checkpoint(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.build_checkpoint_payload()
        payload["boundary_arch_version"] = prod_tcn.infer_boundary_arch_version(
            model_cfg=payload["model_cfg"],
            model_state=payload["model_state"],
        )
        path = output_dir / "boundary_model.pt"
        torch.save(payload, path)
        (output_dir / "train_metrics.json").write_text(
            json.dumps({"history": self.history}, indent=2),
            encoding="utf-8",
        )
        return path


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


class ProbeTrainer:
    def __init__(
        self,
        spec: SpaceSpec,
        *,
        device: torch.device,
        tcn_checkpoint: Path,
        initial_pooler_path: Path,
    ) -> None:
        self.spec = spec
        self.device = device
        self.output_dir = spec.output_dir / "probe"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = prod_probe.ProbePhase1Config(
            epochs=int(spec.cfg.train.stage1_epochs),
            probe_lr=1.5e-5,
            weight_decay=float(spec.cfg.train.weight_decay),
            grad_clip_norm=float(spec.cfg.train.grad_clip_norm),
            chunk_len=32 if str(spec.encoder_model).lower() == "large" else 256,
            chunks_per_stream=int(spec.cfg.train.stage1_chunks_per_stream),
            neg_chunk_fraction=float(spec.cfg.train.stage1_neg_chunk_fraction),
            stage1_last_block_epoch=0,
            stage1_all_epoch=999,
            tcn_tune_mode=str(spec.cfg.train.stage1_tcn_tune_mode),
            tcn_last_blocks=int(spec.cfg.train.stage1_tcn_last_blocks),
            tcn_lr=float(spec.cfg.train.stage1_tcn_lr),
            tcn_weight_decay=float(spec.cfg.train.stage1_tcn_weight_decay),
            stream_sampling_mode=str(spec.cfg.train.stage1_stream_sampling_mode),
            stream_sampling_power=float(spec.cfg.train.stage1_stream_sampling_power),
            stream_sampling_min_weight=float(spec.cfg.train.stage1_stream_sampling_min_weight),
            cyclece_weight=float(spec.cfg.train.stage1_cyclece_weight),
            cyclece_tau=float(spec.cfg.train.cyclece_tau),
            cyclece_radius=int(spec.cfg.train.cyclece_radius),
            smooth_weight=float(spec.cfg.train.stage1_smooth_weight),
            distill_weight=float(spec.cfg.train.stage1_distill_weight),
            class_weight=float(spec.cfg.train.stage1_class_weight),
            fail_if_best_epoch_zero=bool(spec.cfg.train.stage1_fail_if_best_epoch_zero),
            boundary_index_mode=str(spec.cfg.train.boundary_index_mode),
            temporal_structure_mode=str(spec.cfg.temporal_structure_mode),
            seed=int(spec.cfg.train.seed),
        )
        self.history: List[Dict[str, float]] = []
        self.epoch = 0
        self.current_pooler_path = Path(initial_pooler_path).expanduser().resolve()
        self.current_tcn_path = Path(tcn_checkpoint).expanduser().resolve()
        self._feature_cache: Dict[Path, Any] = {}

        self.tcn_payload = torch.load(self.current_tcn_path, map_location="cpu")
        if not isinstance(self.tcn_payload, dict):
            raise RuntimeError(f"Invalid boundary checkpoint payload: {self.current_tcn_path}")
        self.heads = self.tcn_payload.get("heads") if isinstance(self.tcn_payload.get("heads"), (list, tuple)) else []
        self.heads_n = [str(head).strip().lower() for head in self.heads]
        self.combine = "boundary" in self.heads_n and "start" not in self.heads_n and "end" not in self.heads_n
        self.phase_head_indices = prod_probe._resolve_phase_head_indices(self.heads)
        self.use_phase_head = self.phase_head_indices is not None
        multiclass_head = (
            self.tcn_payload.get("multiclass_head")
            if isinstance(self.tcn_payload.get("multiclass_head"), dict)
            else {}
        )
        self.class_head_info = prod_probe._resolve_class_head_info(self.heads, multiclass_head)
        self.class_weights_t = None
        if self.class_head_info is not None and self.class_head_info.get("class_weights") is not None:
            self.class_weights_t = torch.from_numpy(self.class_head_info["class_weights"]).to(self.device)

        self.streams = [
            prod_probe._prepare_stream(
                name=str(record.segment_id),
                token_npz=Path(record.feature_path),
                labels_path=Path(record.label_path),
                ignore_radius=int(self.spec.cfg.train.ignore_radius),
                smooth_sigma=float(self.spec.cfg.train.smooth_sigma),
                boundary_index_mode=str(self.spec.cfg.train.boundary_index_mode),
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
                        "pos_weight_cycle": None,
                        "cycle_weight_trigger_low": 0.35,
                        "cycle_weight_trigger_high": 0.65,
                        "cycle_weight_min": 1.0,
                        "cycle_weight_max": 5.0,
                    },
                )(),
            )
        )
        train_cfg_ckpt = self.tcn_payload.get("train_cfg", {}) if isinstance(self.tcn_payload.get("train_cfg"), dict) else {}
        self.boundary_loss_name = str(train_cfg_ckpt.get("boundary_loss", "focal")).lower()
        self.gamma = float(train_cfg_ckpt.get("gamma", 2.0))
        self.pos_weight_start_end = float(train_cfg_ckpt.get("pos_weight_start_end", 10.0))
        self.phase_loss_name = str(train_cfg_ckpt.get("phase_loss", "mse")).lower()
        self.phase_huber_delta = float(train_cfg_ckpt.get("phase_huber_delta", 0.25))
        self.phase_loss_weight = float(train_cfg_ckpt.get("phase_loss_weight", 0.15))
        self.ignore_radius = int(train_cfg_ckpt.get("ignore_radius", 1))
        self.smooth_sigma = float(train_cfg_ckpt.get("smooth_sigma", 0.0))
        self.left_ctx = int(_resolve_left_context_from_tcn_payload(self.tcn_payload))

        self.probe = _build_probe_with_vendor_classifier(
            int(self.streams[0].tokens.shape[-1]),
            self.device,
            self.current_pooler_path,
            encoder_model=str(self.spec.encoder_model),
            encoder_checkpoint=str(self.spec.encoder_checkpoint),
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
        self.class_weight = float(max(0.0, self.config.class_weight))

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
            "loss_class": 0.0,
            "loss_phase": 0.0,
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
            raw_class_ids = torch.from_numpy(stream.y_class_label_id[start:end]).to(
                device=self.device, dtype=torch.long
            )
            mask_class = torch.from_numpy(stream.mask_class[start:end]).to(device=self.device)
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

            loss_class = logits.new_tensor(0.0)
            if self.class_head_info is not None and self.class_weight > 0.0:
                class_targets = torch.zeros_like(raw_class_ids)
                label_id_to_class_index = self.class_head_info["label_id_to_class_index"]
                for label_id, class_index in label_id_to_class_index.items():
                    class_targets = torch.where(
                        raw_class_ids == int(label_id),
                        torch.full_like(raw_class_ids, int(class_index)),
                        class_targets,
                    )
                logits_class = logits[
                    :, int(self.class_head_info["class_start"]) : int(self.class_head_info["class_stop"])
                ]
                loss_class = prod_tcn._masked_multiclass_ce(
                    logits=logits_class,
                    targets=class_targets,
                    mask=mask_class,
                    class_weights=self.class_weights_t,
                )

            loss_phase = logits.new_tensor(0.0)
            if self.use_phase_head and self.phase_loss_weight > 0.0 and self.phase_head_indices is not None:
                phase_sin_idx = int(self.phase_head_indices[0])
                phase_cos_idx = int(self.phase_head_indices[1])
                y_phase = torch.from_numpy(stream.y_phase[start:end]).to(device=self.device)
                mask_phase = torch.from_numpy(stream.mask_phase[start:end]).to(device=self.device)
                logits_phase = torch.stack([logits[:, phase_sin_idx], logits[:, phase_cos_idx]], dim=-1)
                loss_phase = prod_tcn._phase_loss_masked(
                    pred_phase_logits=logits_phase,
                    target_phase=y_phase,
                    mask_phase=mask_phase,
                    loss_name=str(self.phase_loss_name),
                    huber_delta=float(self.phase_huber_delta),
                )

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
                + (loss_class * float(self.class_weight))
                + (loss_phase * float(self.phase_loss_weight))
                + (loss_cyclece * float(self.config.cyclece_weight))
                + (loss_smooth * float(self.config.smooth_weight))
                + (loss_distill * float(self.config.distill_weight))
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected in probe phase1 space={self.spec.name}")
            loss.backward()
            if float(self.config.grad_clip_norm) > 0:
                params = [param for param in list(self.probe.parameters()) + list(self.tcn.parameters()) if param.requires_grad]
                if params:
                    torch.nn.utils.clip_grad_norm_(params, float(self.config.grad_clip_norm))
            self.optimizer.step()

            totals["loss"] += float(loss.item())
            totals["loss_boundary"] += float(loss_boundary.item())
            totals["loss_start"] += float(loss_start.item())
            totals["loss_end"] += float(loss_end.item())
            totals["loss_cycle"] += float(loss_cycle.item())
            totals["loss_class"] += float(loss_class.item())
            totals["loss_phase"] += float(loss_phase.item())
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
            "Probe space=%s epoch=%d loss=%.4f probe_mode=%s trainable_probe_M=%.3f",
            self.spec.name,
            self.epoch,
            row["loss"],
            probe_mode,
            _count_trainable_params(self.probe) / 1e6,
        )
        return row

    def _encode_tokens_full(self, tokens: np.ndarray) -> np.ndarray:
        outputs: List[np.ndarray] = []
        chunk = int(max(1, DEFAULT_PROBE_EVAL_TOKEN_CHUNK))
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
        logits = prod_tcn._predict_full(
            self.tcn,
            pooled,
            left_ctx=int(self.left_ctx),
            chunk_len=0,
            device=self.device,
        )
        probs = 1.0 / (1.0 + np.exp(-logits[:, :3]))
        return decode_event_pairs(probs, payload["timestamps_ms"], heads=("start", "end", "cycle"))

    def evaluate(self) -> Dict[str, float]:
        self.probe.eval()
        self.tcn.eval()
        predictions = {
            record.segment_id: self._predict_pairs_for_record(record)
            for record in self.spec.split_plan.val_eval_records
        }
        return evaluate_predictions(predictions, records=self.spec.split_plan.val_eval_records)

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
            "temporal_structure_mode": str(self.spec.cfg.temporal_structure_mode),
            "effective_losses": {
                "boundary": True,
                "class": bool(self.class_head_info is not None and self.class_weight > 0.0),
                "cycle": True,
                "phase": bool(self.use_phase_head) and float(self.phase_loss_weight) > 0.0,
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
            "class_weight": float(self.class_weight),
        }
        tcn_ckpt = output_dir / "boundary_model.pt"
        torch.save(tcn_payload, tcn_ckpt)
        (output_dir / "train_metrics.json").write_text(
            json.dumps({"history": self.history}, indent=2),
            encoding="utf-8",
        )
        self.current_pooler_path = probe_ckpt
        self.current_tcn_path = tcn_ckpt
        return probe_ckpt, tcn_ckpt


def _evaluate_trainers(trainers: Mapping[str, Any]) -> EvalBundle:
    metrics_by_space = {name: trainer.evaluate() for name, trainer in trainers.items()}
    return _bundle_metrics(metrics_by_space)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    dst.write_bytes(data)


def _save_best_bundle(
    *,
    stage_name: str,
    output_root: Path,
    tcn_trainers: Optional[Mapping[str, TCNTrainer]] = None,
    probe_trainers: Optional[Mapping[str, ProbeTrainer]] = None,
    eval_bundle: EvalBundle,
) -> None:
    stage_root = output_root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)
    if tcn_trainers is not None:
        for name, trainer in tcn_trainers.items():
            trainer.save_checkpoint(stage_root / SPACE_KEY_BY_NAME[name])
    if probe_trainers is not None:
        for name, trainer in probe_trainers.items():
            trainer.save_checkpoints(stage_root / SPACE_KEY_BY_NAME[name])
    (stage_root / "metrics.json").write_text(
        json.dumps(
            {
                "primary_metric": float(eval_bundle.primary_metric),
                "mean_pair_f1": float(eval_bundle.mean_pair_f1),
                "mean_count_mae": float(eval_bundle.mean_count_mae),
                "mean_start_mae_ms": float(eval_bundle.mean_start_mae_ms),
                "mean_end_mae_ms": float(eval_bundle.mean_end_mae_ms),
                "metrics_by_space": eval_bundle.metrics_by_space,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _print_split_summary(space: SpaceSpec) -> None:
    singleton_cameras = sum(1 for count in space.split_plan.camera_total_counts.values() if int(count) == 1)
    planned_lines = [
        f"{camera_id}:{space.split_plan.camera_val_counts[camera_id]}/{space.split_plan.camera_total_counts[camera_id]}"
        for camera_id in sorted(space.split_plan.camera_total_counts)
    ]
    print(
        f"[split] {space.name} | policy={space.split_plan.split_policy} "
        f"train_videos={len(space.split_plan.train_videos)} "
        f"val_videos={len(space.split_plan.val_videos)} target_val_videos={space.split_plan.target_val_videos:.1f} "
        f"singleton_cameras={singleton_cameras}"
    )
    print(f"[split] {space.name} camera_val_counts={' '.join(planned_lines)}")


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
    output_root = _resolve_output_root() / f"minda_experiment_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    spaces = {name: _build_space_spec(name, output_root) for name in SPACE_ORDER}
    for space in spaces.values():
        _print_split_summary(space)

    print(f"[task] boundary_pairs_dual_space")
    print(f"[workspace_root] {WORKSPACE_ROOT}")
    print(f"[device] {device}")
    print(f"[output_root] {output_root}")

    start_time = time.time()
    peak_vram_mb = 0.0

    # Stage 1: independent TCNs in lockstep.
    tcn_trainers = {name: TCNTrainer(spec, device=device) for name, spec in spaces.items()}
    print(
        "[stage] tcn | duration="
        f"{tcn_stage_seconds:.1f}s | trainable_params_M="
        + " ".join(
            f"{SPACE_KEY_BY_NAME[name]}:{_count_trainable_params(trainer.model) / 1e6:.3f}"
            for name, trainer in tcn_trainers.items()
        )
    )
    TCN_EVAL_EVERY = 10
    best_tcn_eval: Optional[EvalBundle] = None
    tcn_deadline = time.monotonic() + float(tcn_stage_seconds)
    tcn_round = 0
    while True:
        tcn_round += 1
        for name in SPACE_ORDER:
            trainer = tcn_trainers[name]
            trainer.train_epoch()
            peak_vram_mb = max(peak_vram_mb, _peak_vram_mb())
        TCN_MAX_ROUNDS = 80
        past_deadline = time.monotonic() >= tcn_deadline or tcn_round >= TCN_MAX_ROUNDS
        if tcn_round % TCN_EVAL_EVERY == 0 or past_deadline:
            eval_bundle = _evaluate_trainers(tcn_trainers)
            print(
                f"[tcn] round={tcn_round} | primary={eval_bundle.primary_metric:.6f} "
                f"mean_pair_f1={eval_bundle.mean_pair_f1:.6f} mean_count_mae={eval_bundle.mean_count_mae:.6f}"
            )
            if _should_replace_eval(best_tcn_eval, eval_bundle):
                best_tcn_eval = eval_bundle
                _save_best_bundle(
                    stage_name="tcn_best",
                    output_root=output_root,
                    tcn_trainers=tcn_trainers,
                    eval_bundle=eval_bundle,
                )
        if past_deadline:
            break

    if best_tcn_eval is None:
        raise RuntimeError("No TCN evaluation bundle was produced.")

    tcn_best_root = output_root / "tcn_best"
    probe_trainers = {}
    for name, spec in spaces.items():
        probe_trainers[name] = ProbeTrainer(
            spec,
            device=device,
            tcn_checkpoint=tcn_best_root / SPACE_KEY_BY_NAME[name] / "boundary_model.pt",
            initial_pooler_path=spec.initial_pooler_path,
        )

    print(
        "[stage] probe_phase1 | duration="
        f"{probe_stage_seconds:.1f}s | trainable_params_M="
        + " ".join(
            f"{SPACE_KEY_BY_NAME[name]}:{_count_trainable_params(trainer.probe) / 1e6:.3f}"
            for name, trainer in probe_trainers.items()
        )
    )
    best_probe_eval: Optional[EvalBundle] = None
    tcn_elapsed = time.time() - start_time
    remaining_budget = max(60.0, float(total_budget_seconds) - tcn_elapsed)
    probe_deadline = time.monotonic() + remaining_budget
    print(f"[probe] budget={remaining_budget:.1f}s (tcn used {tcn_elapsed:.1f}s)")
    probe_round = 0
    while True:
        probe_round += 1
        for name in SPACE_ORDER:
            trainer = probe_trainers[name]
            trainer.train_epoch()
            peak_vram_mb = max(peak_vram_mb, _peak_vram_mb())
        eval_bundle = _evaluate_trainers(probe_trainers)
        print(
            f"[probe] round={probe_round} | primary={eval_bundle.primary_metric:.6f} "
            f"mean_pair_f1={eval_bundle.mean_pair_f1:.6f} mean_count_mae={eval_bundle.mean_count_mae:.6f}"
        )
        if _should_replace_eval(best_probe_eval, eval_bundle):
            best_probe_eval = eval_bundle
            _save_best_bundle(
                stage_name="probe_best",
                output_root=output_root,
                probe_trainers=probe_trainers,
                eval_bundle=eval_bundle,
            )
        if time.monotonic() >= probe_deadline:
            break

    if best_probe_eval is None:
        raise RuntimeError("No probe evaluation bundle was produced.")

    total_seconds = float(time.time() - start_time)
    summary = {
        "model_family": MODEL_FAMILY,
        "task_mode": "boundary_pairs_dual_space",
        "time_budget_seconds": float(total_budget_seconds),
        "tcn_stage_seconds": float(tcn_stage_seconds),
        "probe_stage_seconds": float(probe_stage_seconds),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "primary_metric": float(best_probe_eval.primary_metric),
        "mean_pair_f1": float(best_probe_eval.mean_pair_f1),
        "mean_count_mae": float(best_probe_eval.mean_count_mae),
        "mean_start_mae_ms": float(best_probe_eval.mean_start_mae_ms),
        "mean_end_mae_ms": float(best_probe_eval.mean_end_mae_ms),
        "metrics_by_space": best_probe_eval.metrics_by_space,
        "spaces": {
            name: {
                "source_run_dir": str(spec.source_run_dir),
                "cache_root": str(spec.cache_root),
                "split_policy": str(spec.split_plan.split_policy),
                "train_segments": len(spec.split_plan.train_records),
                "val_segments": len(spec.split_plan.val_records),
                "val_eval_segments": len(spec.split_plan.val_eval_records),
                "train_videos": len(spec.split_plan.train_videos),
                "val_videos": len(spec.split_plan.val_videos),
            }
            for name, spec in spaces.items()
        },
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if total_seconds > float(total_timeout_seconds):
        print("FAIL: exceeded total timeout")
        return 1

    print("---")
    print(f"val_pair_f1:        {best_probe_eval.primary_metric:.6f}")
    print(f"val_pair_f1_mean:   {best_probe_eval.mean_pair_f1:.6f}")
    print(f"val_count_mae:      {best_probe_eval.mean_count_mae:.6f}")
    print(f"val_start_mae_ms:   {best_probe_eval.mean_start_mae_ms:.1f}")
    print(f"val_end_mae_ms:     {best_probe_eval.mean_end_mae_ms:.1f}")
    for name in SPACE_ORDER:
        metrics = best_probe_eval.metrics_by_space[name]
        key = SPACE_KEY_BY_NAME[name]
        print(f"{key}_val_pair_f1:   {metrics['val_pair_f1']:.6f}")
        print(f"{key}_val_count_mae: {metrics['val_count_mae']:.6f}")
        print(f"{key}_val_start_mae_ms:{metrics['val_start_mae_ms']:.1f}")
        print(f"{key}_val_end_mae_ms:{metrics['val_end_mae_ms']:.1f}")
    print(f"training_seconds:   {tcn_stage_seconds + probe_stage_seconds:.1f}")
    print(f"total_seconds:      {total_seconds:.1f}")
    print(f"time_budget_seconds:{total_budget_seconds:.1f}")
    print(f"tcn_stage_seconds:  {tcn_stage_seconds:.1f}")
    print(f"probe_stage_seconds:{probe_stage_seconds:.1f}")
    print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
    print(f"num_rounds_tcn:     {tcn_round}")
    print(f"num_rounds_probe:   {probe_round}")
    print(f"tcn_trainable_params_M: {sum(_count_trainable_params(trainer.model) for trainer in tcn_trainers.values()) / 1e6:.3f}")
    print(f"pooler_trainable_params_M: {sum(_count_trainable_params(trainer.probe) for trainer in probe_trainers.values()) / 1e6:.3f}")
    print(f"model_family:       {MODEL_FAMILY}")
    print(f"task_mode:          boundary_pairs_dual_space")
    print(f"pooler_tune_mode:   phase1_per_space")
    print(f"representation_mode:pooled_z0_then_tokens")
    print(f"cache_root:         button={spaces['minda-button-tcn'].cache_root} subassembly={spaces['minda-subassembly-tcn'].cache_root}")
    print(f"train_segments:     {sum(len(spec.split_plan.train_records) for spec in spaces.values())}")
    print(f"val_segments:       {sum(len(spec.split_plan.val_records) for spec in spaces.values())}")
    print(f"cache_version:      {CACHE_VERSION}")
    print(f"output_root_final:  {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
