"""
Exact-prod V2 phase-1 autoresearch runner for Mangalam packing.

This branch intentionally fixes everything except the phase-1 pooler finetune:

- exact Mangalam packing dense-temporal cache
- deterministic 60/40 camera-stratified validation split
- fixed pregenerated prod `reduced_rf_halo16` stage-0 checkpoint
- exact prod V2 phase-1 epoch-eval harness
- primary metric = halo16 pair F1 on the held-out validation split

`train.py` is still the mutable research surface, but the baseline is a thin
wrapper around the production phase-1 runner so the starting point matches prod
as closely as possible.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.cache_contract import (
    CACHE_VERSION,
    DEFAULT_SPLIT_POLICY,
    PAIR_TOLERANCE_MS,
    TIME_BUDGET,
    TOTAL_TIMEOUT_SECONDS,
    VAL_RATIO,
    SegmentRecord,
    configure_cache_paths,
    load_manifest,
    load_split_records,
)

LOGGER = logging.getLogger("autoresearch.mangalam.phase1")

MODEL_FAMILY = "mangalam_phase1_prod6040_halo16_fixed_stage0"
TASK_MODE = "pooler_phase1_single_space_fixed_stage0"
SPACE_NAME = "mangalam-packing"
SPACE_ID = "1e2544ee-b324-4c86-8453-e1fd7fad9c04"
SOURCE_RUN_ID = "9713de5f-df45-49b7-9b41-24fb768a6325"
SOURCE_RUN_NUMBER = 10
SEED = 42

DEFAULT_CACHE_BASE_DIR = Path("/tmp/autoresearch-mangalam-packing-cache")
DEFAULT_OUTPUT_ROOT = Path("/tmp/autoresearch_mangalam_phase1_runs")
DEFAULT_FIXED_STAGE0_DIRNAME = "fixed_stage0_prod_reduced_rf_halo16"
DEFAULT_FIXED_STAGE0_VARIANT = "baseline"
DEFAULT_TCN_STAGE_SECONDS = 0.0
DEFAULT_PROBE_STAGE_SECONDS = 300.0

PROD_PHASE1_EPOCHS = 20
PROD_PROBE_LR = 2.6e-5
PROD_HALO = 16
PROD_CENTER_CHUNK_LEN = 32
PROD_CHUNK32_SIZE = 32
PROD_THRESHOLD_PROXY_PROB = 0.15
PROD_THRESHOLD_PROXY_TOLERANCE = 2

EXPECTED_STAGE0_MODEL_CFG = {
    "input_dim": None,
    "hidden_dim": 128,
    "kernel_size": 5,
    "dropout": 0.1,
    "use_layernorm": True,
    "dilations": (1, 2, 4),
    "bidirectional": True,
    "task_specific_heads": True,
    "base_heads": 3,
}

EXPECTED_STAGE0_TRAIN_CFG = {
    "boundary_loss": "focal",
    "gamma": 2.0,
    "pos_weight_start_end": 10.0,
    "pos_weight_cycle": 1.0,
    "ignore_radius": 1,
    "smooth_sigma": 0.0,
    "combine_start_end": False,
    "boundary_index_mode": "ordered_nearest",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 200,
    "chunk_len": 256,
    "chunks_per_epoch": 64,
    "neg_chunk_fraction": 0.40,
    "grad_clip_norm": 1.0,
    "stage1_enabled": True,
    "stage1_epochs": 20,
    "stage1_probe_lr": 2.6e-5,
    "stage1_last_block_epoch": 0,
    "stage1_all_epoch": 999,
    "stage1_chunks_per_stream": 12,
    "stage1_neg_chunk_fraction": 0.25,
    "stage1_tcn_tune_mode": "frozen",
    "stage1_tcn_last_blocks": 1,
    "stage1_tcn_lr": 5e-5,
    "stage1_tcn_weight_decay": 1e-4,
    "stage1_stream_sampling_mode": "uniform",
    "stage1_stream_sampling_power": 0.5,
    "stage1_stream_sampling_min_weight": 1.0,
    "stage1_cyclece_weight": 1.0,
    "stage1_smooth_weight": 0.05,
    "stage1_distill_weight": 0.10,
    "stage1_class_weight": 0.0,
    "stage1_fail_if_best_epoch_zero": False,
}


def _resolve_workspace_root() -> Path:
    candidates: List[Path] = []
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


@dataclass(frozen=True)
class CacheSpec:
    cache_root: Path
    manifest: Dict[str, Any]
    train_records: Tuple[SegmentRecord, ...]
    val_eval_records: Tuple[SegmentRecord, ...]
    pooler_path: Path
    encoder_model: str
    encoder_checkpoint: Path
    train_segments_path: Path
    val_eval_segments_path: Path


def _resolve_output_root() -> Path:
    override = os.getenv("AUTORESEARCH_OUTPUT_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT.resolve()


def _resolve_cache_base_dir() -> Path:
    override = os.getenv("AUTORESEARCH_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CACHE_BASE_DIR.resolve()


def _resolve_cache_root() -> Path:
    return _resolve_cache_base_dir() / CACHE_VERSION


def _resolve_fixed_stage0_checkpoint(cache_root: Path) -> Path:
    override = os.getenv("AUTORESEARCH_FIXED_STAGE0_CHECKPOINT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return cache_root.parent / DEFAULT_FIXED_STAGE0_DIRNAME / "boundary_model.pt"


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


def _peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _resolve_pooler_path(records: Sequence[SegmentRecord]) -> Path:
    for record in records:
        raw = str(record.pooler_checkpoint or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    fallback = WORKSPACE_ROOT / "encoder_models" / "vjepa2_attention_poolers" / "ssv2-vitl-16x2x3.pt"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("No valid pooler checkpoint found in the prepared Mangalam cache.")


def _resolve_encoder_spec(token_dim: int) -> Tuple[str, Path]:
    token_dim_i = int(token_dim)
    if token_dim_i == 1024:
        model = "large"
        checkpoint = WORKSPACE_ROOT / "encoder_models" / "vitl.pt"
    elif token_dim_i == 1408:
        model = "giant"
        checkpoint = WORKSPACE_ROOT / "encoder_models" / "vitg-384.pt"
    else:
        raise RuntimeError(f"Unsupported token_dim={token_dim_i} for V-JEPA pooler replay")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {checkpoint}")
    return model, checkpoint.resolve()


def _load_cache_spec(cache_root: Path) -> CacheSpec:
    configure_cache_paths(cache_root)
    manifest = load_manifest(cache_root=cache_root)
    split_policy = str(manifest.get("split_policy") or "").strip()
    val_ratio = float(manifest.get("val_ratio") or 0.0)
    if split_policy != DEFAULT_SPLIT_POLICY:
        raise RuntimeError(
            f"Cache {cache_root} uses split_policy={split_policy!r}, expected {DEFAULT_SPLIT_POLICY!r}"
        )
    if not math.isclose(val_ratio, float(VAL_RATIO), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Cache {cache_root} uses val_ratio={val_ratio}, expected {VAL_RATIO}")

    train_records = tuple(load_split_records("train", cache_root=cache_root))
    val_eval_records = tuple(load_split_records("val_eval", cache_root=cache_root))
    if not train_records or not val_eval_records:
        raise RuntimeError(f"Cache {cache_root} has an empty train/val_eval split")

    pooler_path = _resolve_pooler_path(train_records + val_eval_records)
    encoder_model, encoder_checkpoint = _resolve_encoder_spec(int(train_records[0].token_dim))
    train_segments_path = cache_root / "index" / "train_segments.jsonl"
    val_eval_segments_path = cache_root / "index" / "val_eval_segments.jsonl"
    if not train_segments_path.exists() or not val_eval_segments_path.exists():
        raise FileNotFoundError(f"Cache {cache_root} is missing train/val_eval jsonl files")

    return CacheSpec(
        cache_root=cache_root,
        manifest=dict(manifest),
        train_records=train_records,
        val_eval_records=val_eval_records,
        pooler_path=pooler_path,
        encoder_model=encoder_model,
        encoder_checkpoint=encoder_checkpoint,
        train_segments_path=train_segments_path.resolve(),
        val_eval_segments_path=val_eval_segments_path.resolve(),
    )


def _normalize_dilations(value: Any) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return tuple()
    return tuple(int(item) for item in value)


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise RuntimeError(f"Fixed stage-0 checkpoint has {field}={actual!r}, expected {expected!r}")


def _require_close(actual: Any, expected: float, *, field: str) -> None:
    actual_f = float(actual)
    if not math.isclose(actual_f, float(expected), rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Fixed stage-0 checkpoint has {field}={actual_f}, expected {expected}")


def _validate_fixed_stage0_checkpoint(payload: Mapping[str, Any]) -> None:
    model_cfg = payload.get("model_cfg")
    train_cfg = payload.get("train_cfg")
    heads = payload.get("heads")
    if not isinstance(model_cfg, Mapping) or not isinstance(train_cfg, Mapping):
        raise RuntimeError("Fixed stage-0 checkpoint is missing model_cfg/train_cfg")
    _require_equal(str(payload.get("seq_model") or "").strip().lower(), "tcn", field="seq_model")
    if not isinstance(heads, list):
        raise RuntimeError("Fixed stage-0 checkpoint is missing heads")
    _require_equal(tuple(str(item).strip().lower() for item in heads[:3]), ("start", "end", "cycle"), field="heads[:3]")

    _require_equal(int(model_cfg.get("hidden_dim", -1)), int(EXPECTED_STAGE0_MODEL_CFG["hidden_dim"]), field="model_cfg.hidden_dim")
    _require_equal(int(model_cfg.get("kernel_size", -1)), int(EXPECTED_STAGE0_MODEL_CFG["kernel_size"]), field="model_cfg.kernel_size")
    _require_close(float(model_cfg.get("dropout", -1.0)), float(EXPECTED_STAGE0_MODEL_CFG["dropout"]), field="model_cfg.dropout")
    _require_equal(bool(model_cfg.get("use_layernorm")), bool(EXPECTED_STAGE0_MODEL_CFG["use_layernorm"]), field="model_cfg.use_layernorm")
    _require_equal(_normalize_dilations(model_cfg.get("dilations")), EXPECTED_STAGE0_MODEL_CFG["dilations"], field="model_cfg.dilations")
    _require_equal(bool(model_cfg.get("bidirectional") or model_cfg.get("tcn_bidirectional")), True, field="model_cfg.bidirectional")
    _require_equal(bool(model_cfg.get("task_specific_heads") or model_cfg.get("tcn_task_specific_heads")), True, field="model_cfg.task_specific_heads")

    for field, expected in EXPECTED_STAGE0_TRAIN_CFG.items():
        actual = train_cfg.get(field)
        if isinstance(expected, float):
            _require_close(actual, expected, field=f"train_cfg.{field}")
        else:
            _require_equal(actual, expected, field=f"train_cfg.{field}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _prepare_stage0_wrapper(
    *,
    output_root: Path,
    cache_spec: CacheSpec,
    fixed_stage0_checkpoint: Path,
) -> Tuple[Path, Path]:
    wrapper_root = output_root / "fixed_stage0_wrapper"
    baseline_dir = wrapper_root / DEFAULT_FIXED_STAGE0_VARIANT
    if wrapper_root.exists():
        shutil.rmtree(wrapper_root)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    wrapped_checkpoint = baseline_dir / "boundary_model.pt"
    checkpoint_payload = torch.load(fixed_stage0_checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError(f"Invalid fixed stage-0 checkpoint payload: {fixed_stage0_checkpoint}")
    inference_payload = dict(checkpoint_payload.get("inference") or {})
    inference_payload["encoder_model"] = str(cache_spec.encoder_model)
    inference_payload["encoder_checkpoint"] = str(cache_spec.encoder_checkpoint)
    checkpoint_payload["inference"] = inference_payload
    torch.save(checkpoint_payload, wrapped_checkpoint)

    summary_payload = {
        "space_name": SPACE_NAME,
        "space_id": SPACE_ID,
        "base_run_id": SOURCE_RUN_ID,
        "run_number": int(SOURCE_RUN_NUMBER),
        "cache_root": str(cache_spec.cache_root),
        "encoder_model": str(cache_spec.encoder_model),
        "encoder_checkpoint": str(cache_spec.encoder_checkpoint),
        "pooler_path": str(cache_spec.pooler_path),
        "stage0_variant": DEFAULT_FIXED_STAGE0_VARIANT,
        "source_fixed_stage0_checkpoint": str(fixed_stage0_checkpoint),
        "source_run_dirs": list(cache_spec.manifest.get("source_run_dirs") or []),
        "split_policy": str(cache_spec.manifest.get("split_policy") or DEFAULT_SPLIT_POLICY),
        "val_ratio": float(cache_spec.manifest.get("val_ratio") or VAL_RATIO),
        "seed": int(cache_spec.manifest.get("seed") or SEED),
        "train_segments_file": str(cache_spec.train_segments_path),
        "val_segments_file": str(cache_spec.val_eval_segments_path),
        "no_finalize": True,
        "no_upload": True,
        "no_supabase": True,
    }
    _write_json(wrapper_root / "summary.json", summary_payload)

    metadata_payload = {
        "branch_contract": "mangalam_phase1_prod6040_halo16",
        "space_name": SPACE_NAME,
        "space_id": SPACE_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_run_number": int(SOURCE_RUN_NUMBER),
        "cache_version": CACHE_VERSION,
        "split_policy": str(cache_spec.manifest.get("split_policy") or DEFAULT_SPLIT_POLICY),
        "val_ratio": float(cache_spec.manifest.get("val_ratio") or VAL_RATIO),
        "pair_tolerance_ms": int(PAIR_TOLERANCE_MS),
        "fixed_stage0_checkpoint": str(fixed_stage0_checkpoint),
        "checkpoint_kind": "fixed_prod_reduced_rf_halo16",
    }
    metadata_path = wrapper_root / "metadata_stamp.json"
    _write_json(metadata_path, metadata_payload)
    return wrapper_root, metadata_path


def _build_prod_phase1_args(
    *,
    stage0_wrapper_root: Path,
    cache_spec: CacheSpec,
    output_root: Path,
    metadata_stamp_path: Path,
    device: str,
) -> List[str]:
    return [
        "--stage0-run-root",
        str(stage0_wrapper_root),
        "--cache-root",
        str(cache_spec.cache_root),
        "--output-root",
        str(output_root),
        "--variant",
        DEFAULT_FIXED_STAGE0_VARIANT,
        "--train-segments-file",
        str(cache_spec.train_segments_path),
        "--val-segments-file",
        str(cache_spec.val_eval_segments_path),
        "--device",
        str(device),
        "--seed",
        str(int(SEED)),
        "--epochs",
        str(int(PROD_PHASE1_EPOCHS)),
        "--probe-lr",
        str(float(PROD_PROBE_LR)),
        "--threshold-proxy-prob",
        str(float(PROD_THRESHOLD_PROXY_PROB)),
        "--threshold-proxy-tolerance",
        str(int(PROD_THRESHOLD_PROXY_TOLERANCE)),
        "--halo",
        str(int(PROD_HALO)),
        "--center-chunk-len",
        str(int(PROD_CENTER_CHUNK_LEN)),
        "--chunk32-size",
        str(int(PROD_CHUNK32_SIZE)),
        "--metadata-stamp-path",
        str(metadata_stamp_path),
    ]


def _halo_selection_tuple(payload: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    halo_eval = payload["halo16_eval"]
    halo_metrics = halo_eval["metrics"]
    proxy_eval = payload["threshold_proxy_eval"]
    return (
        float(halo_metrics.get("val_pair_f1", 0.0)),
        -float(proxy_eval.get("val_proxy_total_false_count", 1e18)),
        -float(halo_metrics.get("val_count_mae", 1e18)),
        -float(halo_eval.get("timing_mae_ms", 1e18)),
    )


def _run_prod_phase1_epoch_eval(*, args: Sequence[str]) -> int:
    script_path = WORKSPACE_ROOT / "tmp_scripts" / "run_minda_reduced_halo16_prod_phase1_epoch_eval.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing prod phase-1 script: {script_path}")
    module_name = f"_autoresearch_prod_phase1_epoch_eval_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load prod phase-1 module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    saved_argv = list(sys.argv)
    try:
        spec.loader.exec_module(module)
        sys.argv = [str(script_path), *list(args)]
        result = module.main()
    finally:
        sys.argv = saved_argv
        sys.modules.pop(module_name, None)
    return int(result or 0)


def _prime_vjepa_runtime_env(cache_spec: CacheSpec) -> None:
    encoder_variant = "vit_large" if str(cache_spec.encoder_model).strip().lower() == "large" else "vit_giant"
    os.environ.setdefault("VJEPA_ENCODER_MODEL", encoder_variant)
    os.environ.setdefault("VJEPA_ENCODER_CHECKPOINT", str(cache_spec.encoder_checkpoint))
    os.environ.setdefault("DENSE_TEMPORAL_ENCODER_MODEL", str(cache_spec.encoder_model))
    os.environ.setdefault("DENSE_TEMPORAL_ENCODER_CHECKPOINT", str(cache_spec.encoder_checkpoint))
    os.environ.setdefault("DENSE_TEMPORAL_POOLER_PATH", str(cache_spec.pooler_path))


def _extract_best_metrics(prod_summary: Mapping[str, Any]) -> Dict[str, Any]:
    best = prod_summary["best_by_halo16"]
    best_metrics = best["metrics"]
    baseline = prod_summary["baseline"]
    return {
        "best_epoch": int(best["epoch"]),
        "baseline_halo16_pair_f1": float(baseline["halo16_eval"]["metrics"]["val_pair_f1"]),
        "val_halo16_pair_f1": float(best_metrics["halo16_eval"]["metrics"]["val_pair_f1"]),
        "val_legacy_pair_f1": float(best_metrics["legacy_prod_eval"]["metrics"]["val_pair_f1"]),
        "val_chunk32_pair_f1": float(best_metrics["chunk32_eval"]["metrics"]["val_pair_f1"]),
        "val_proxy_macro_f1": float(best_metrics["threshold_proxy_eval"]["val_proxy_macro_f1"]),
        "val_proxy_false": int(round(float(best_metrics["threshold_proxy_eval"]["val_proxy_total_false_count"]))),
        "val_count_mae": float(best_metrics["halo16_eval"]["metrics"]["val_count_mae"]),
        "val_timing_mae_ms": float(best_metrics["halo16_eval"]["timing_mae_ms"]),
    }


def _print_summary(
    *,
    metrics: Mapping[str, Any],
    training_seconds: float,
    total_seconds: float,
    total_budget_seconds: float,
    tcn_stage_seconds: float,
    probe_stage_seconds: float,
    peak_vram_mb: float,
) -> None:
    print("---")
    print(f"val_halo16_pair_f1:{float(metrics['val_halo16_pair_f1']):.6f}")
    print(f"val_legacy_pair_f1:{float(metrics['val_legacy_pair_f1']):.6f}")
    print(f"val_chunk32_pair_f1:{float(metrics['val_chunk32_pair_f1']):.6f}")
    print(f"val_proxy_macro_f1:{float(metrics['val_proxy_macro_f1']):.6f}")
    print(f"val_proxy_false:   {int(metrics['val_proxy_false'])}")
    print(f"val_count_mae:     {float(metrics['val_count_mae']):.6f}")
    print(f"val_timing_mae_ms: {float(metrics['val_timing_mae_ms']):.1f}")
    print(f"baseline_halo16_pair_f1:{float(metrics['baseline_halo16_pair_f1']):.6f}")
    print(f"best_epoch:        {int(metrics['best_epoch'])}")
    print(f"training_seconds:  {float(training_seconds):.1f}")
    print(f"total_seconds:     {float(total_seconds):.1f}")
    print(f"time_budget_seconds:{float(total_budget_seconds):.1f}")
    print(f"tcn_stage_seconds: {float(tcn_stage_seconds):.1f}")
    print(f"probe_stage_seconds:{float(probe_stage_seconds):.1f}")
    print(f"peak_vram_mb:      {float(peak_vram_mb):.1f}")
    print(f"cache_version:     {CACHE_VERSION}")
    print(f"model_family:      {MODEL_FAMILY}")
    print(f"task_mode:         {TASK_MODE}")
    print("pooler_tune_mode:  phase1_prod_v2_epoch_eval")
    print("representation_mode:tokens_to_pooler_fixed_stage0")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_budget_seconds = resolve_time_budget_seconds()
    total_timeout_seconds = resolve_total_timeout_seconds()
    tcn_stage_seconds, probe_stage_seconds = resolve_stage_budget_seconds(total_budget_seconds)
    if tcn_stage_seconds > 0.0:
        LOGGER.warning(
            "This branch is phase-1-only. Ignoring requested TCN budget %.1fs and keeping the fixed stage-0 checkpoint.",
            float(tcn_stage_seconds),
        )
        tcn_stage_seconds = 0.0

    cache_root = _resolve_cache_root()
    cache_spec = _load_cache_spec(cache_root)
    _prime_vjepa_runtime_env(cache_spec)

    fixed_stage0_checkpoint = _resolve_fixed_stage0_checkpoint(cache_root)
    if not fixed_stage0_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing fixed prod stage-0 checkpoint: {fixed_stage0_checkpoint}. "
            "Pregenerate the Mangalam 60/40 reduced_rf_halo16 baseline first or set "
            "AUTORESEARCH_FIXED_STAGE0_CHECKPOINT."
        )
    payload = torch.load(fixed_stage0_checkpoint, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid fixed stage-0 checkpoint payload: {fixed_stage0_checkpoint}")
    _validate_fixed_stage0_checkpoint(payload)

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_root = _resolve_output_root() / f"mangalam_phase1_prod6040_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    stage0_wrapper_root, metadata_path = _prepare_stage0_wrapper(
        output_root=output_root,
        cache_spec=cache_spec,
        fixed_stage0_checkpoint=fixed_stage0_checkpoint,
    )
    phase1_output_root = output_root / "phase1_prod_epoch_eval"
    phase1_args = _build_prod_phase1_args(
        stage0_wrapper_root=stage0_wrapper_root,
        cache_spec=cache_spec,
        output_root=phase1_output_root,
        metadata_stamp_path=metadata_path,
        device=device,
    )

    LOGGER.info(
        "Launching exact prod phase-1 for %s cache=%s fixed_stage0=%s timeout=%.1fs",
        SPACE_NAME,
        cache_spec.cache_root,
        fixed_stage0_checkpoint,
        float(total_timeout_seconds),
    )
    print(f"[space] {SPACE_NAME}")
    print(f"[cache_root] {cache_spec.cache_root}")
    print(f"[fixed_stage0_checkpoint] {fixed_stage0_checkpoint}")
    print(f"[output_root] {output_root}")
    print(f"[device] {device}")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return_code = _run_prod_phase1_epoch_eval(args=phase1_args)
    if return_code != 0:
        raise RuntimeError(f"Prod phase-1 epoch-eval runner returned {return_code}")
    peak_vram_mb = _peak_vram_mb()
    total_seconds = float(time.time() - start_time)
    if total_seconds > float(total_timeout_seconds):
        raise RuntimeError(
            f"Prod phase-1 epoch-eval exceeded timeout: total_seconds={total_seconds:.1f} "
            f"timeout_seconds={float(total_timeout_seconds):.1f}"
        )

    prod_summary_path = phase1_output_root / "summary.json"
    if not prod_summary_path.exists():
        raise FileNotFoundError(f"Missing prod phase-1 summary: {prod_summary_path}")
    prod_summary = json.loads(prod_summary_path.read_text(encoding="utf-8"))
    metrics = _extract_best_metrics(prod_summary)
    training_seconds = float(prod_summary.get("elapsed_seconds", total_seconds))

    run_summary = {
        "model_family": MODEL_FAMILY,
        "task_mode": TASK_MODE,
        "space_name": SPACE_NAME,
        "space_id": SPACE_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_run_number": int(SOURCE_RUN_NUMBER),
        "cache_root": str(cache_spec.cache_root),
        "cache_version": CACHE_VERSION,
        "split_policy": str(cache_spec.manifest.get("split_policy") or DEFAULT_SPLIT_POLICY),
        "val_ratio": float(cache_spec.manifest.get("val_ratio") or VAL_RATIO),
        "train_segments": len(cache_spec.train_records),
        "val_eval_segments": len(cache_spec.val_eval_records),
        "fixed_stage0_checkpoint": str(fixed_stage0_checkpoint),
        "phase1_output_root": str(phase1_output_root),
        "prod_summary_path": str(prod_summary_path),
        "time_budget_seconds": float(total_budget_seconds),
        "tcn_stage_seconds": float(tcn_stage_seconds),
        "probe_stage_seconds": float(probe_stage_seconds),
        "training_seconds": float(training_seconds),
        "total_seconds": float(total_seconds),
        "peak_vram_mb": float(peak_vram_mb),
        "best_metrics": metrics,
        "best_by_halo16": prod_summary.get("best_by_halo16"),
        "baseline": prod_summary.get("baseline"),
        "final_artifacts": prod_summary.get("final_artifacts"),
    }
    _write_json(output_root / "run_summary.json", run_summary)

    _print_summary(
        metrics=metrics,
        training_seconds=training_seconds,
        total_seconds=total_seconds,
        total_budget_seconds=total_budget_seconds,
        tcn_stage_seconds=tcn_stage_seconds,
        probe_stage_seconds=probe_stage_seconds,
        peak_vram_mb=peak_vram_mb,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
