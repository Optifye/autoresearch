"""
One-time cache preparation for dense-temporal autoresearch experiments.

The cache is built from existing dense-temporal feature/label artifacts rather
than raw video extraction. This keeps the repo faithful to the original
autoresearch contract:

- `prepare.py` is the fixed harness.
- `train.py` is the mutable research surface.
- the frozen V-JEPA encoder is never rerun during experiments.

Usage:
    python prepare.py
    python prepare.py --run-id <dense_temporal_run_id>
    python prepare.py --source-run-dir /tmp/embedding_runs/<run>/dense_temporal
    python prepare.py --camera-include-regex "onemed"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import struct
import zipfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .contracts import parse_dense_temporal_snapshot
from .decode_start_end_pairs import (
    StartEndPairDecodeConfig,
    decode_start_end_pairs,
)
from .dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Fixed experiment contract
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

TIME_BUDGET = 600
TOTAL_TIMEOUT_SECONDS = 900
PAIR_TOLERANCE_MS = 1000
VAL_RATIO = 0.5
CACHE_VERSION = "onemed_dense_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _workspace_root_candidates() -> List[Path]:
    out: List[Path] = []
    seen: set[str] = set()
    for raw in (
        os.getenv("AUTORESEARCH_WORKSPACE_ROOT", "").strip(),
        "/workspace",
    ):
        if not raw:
            continue
        path = Path(raw).expanduser()
        key = str(path.resolve() if path.exists() else path)
        if key in seen:
            continue
        out.append(path.resolve() if path.exists() else path)
        seen.add(key)
    return out


def _resolve_cache_dir() -> Path:
    override = os.getenv("AUTORESEARCH_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve() / CACHE_VERSION
    return Path(os.path.expanduser("~")) / ".cache" / "autoresearch" / CACHE_VERSION


def configure_cache_paths(cache_root: Path) -> None:
    global CACHE_DIR
    global RAW_FEATURES_DIR
    global MATERIALIZED_RUNS_DIR
    global POOLERS_DIR
    global SPLITS_DIR
    global INDEX_DIR
    global MANIFEST_PATH
    global TRAIN_SEGMENTS_PATH
    global VAL_SEGMENTS_PATH
    global VAL_EVAL_SEGMENTS_PATH

    CACHE_DIR = cache_root
    RAW_FEATURES_DIR = CACHE_DIR / "raw_features"
    MATERIALIZED_RUNS_DIR = CACHE_DIR / "materialized_runs"
    POOLERS_DIR = CACHE_DIR / "poolers"
    SPLITS_DIR = CACHE_DIR / "splits"
    INDEX_DIR = CACHE_DIR / "index"
    MANIFEST_PATH = CACHE_DIR / "manifest.json"
    TRAIN_SEGMENTS_PATH = INDEX_DIR / "train_segments.jsonl"
    VAL_SEGMENTS_PATH = INDEX_DIR / "val_segments.jsonl"
    VAL_EVAL_SEGMENTS_PATH = INDEX_DIR / "val_eval_segments.jsonl"


configure_cache_paths(_resolve_cache_dir())

DEFAULT_SOURCE_GLOBS = (
    "/tmp/embedding_runs/*/dense_temporal",
)

DEFAULT_DECODE_CONFIG = StartEndPairDecodeConfig(
    candidate_min_prob=0.15,
    peak_min_distance_s=0.0,
    min_pair_s=0.0,
    max_pair_s=None,
    min_gap_s=0.0,
    pair_score_bias=0.0,
    cycle_weight=0.5,
    outside_pad_s=1.0,
    allow_touching_pairs=False,
    objective="count_then_score",
)


@dataclass(frozen=True)
class EventPair:
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SegmentRecord:
    segment_id: str
    split: str
    video_id: str
    camera_id: str
    source_run_dir: str
    feature_path: str
    label_path: str
    pooler_checkpoint: str
    pooler_sha: str
    embedding_dim: int
    token_dim: int
    tokens_per_window: int
    num_total_windows: int
    fps: float
    supervised_start_ms: int
    supervised_end_ms: int
    supervised_start_idx: int
    supervised_end_idx: int
    eval_start_ms: Optional[int]
    eval_end_ms: Optional[int]
    eval_start_idx: Optional[int]
    eval_end_idx: Optional[int]
    event_pairs_ms: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class MatchResult:
    matched_pairs: int
    total_cost_ms: int
    assignment: Tuple[Tuple[int, int], ...]


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON type: {type(value)!r}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_workspace_path(raw: Optional[str]) -> Optional[Path]:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.startswith("/workspace/"):
        suffix = text[len("/workspace/") :]
        for root in _workspace_root_candidates():
            candidate = (root / suffix).expanduser()
            if candidate.exists():
                return candidate.resolve()
        roots = _workspace_root_candidates()
        if roots:
            return (roots[0] / suffix).expanduser()
    path = Path(text).expanduser()
    return path.resolve() if path.exists() else path


def _stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _ensure_dirs(cache_root: Path) -> None:
    for path in (cache_root, MATERIALIZED_RUNS_DIR, RAW_FEATURES_DIR, POOLERS_DIR, SPLITS_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)


_PREPARE_ENV_LOADED = False


def _load_repo_env_once() -> None:
    global _PREPARE_ENV_LOADED
    if _PREPARE_ENV_LOADED:
        return
    env_paths: List[Path] = []
    explicit_env = os.getenv("AUTORESEARCH_ENV_PATH", "").strip()
    if explicit_env:
        env_paths.append(Path(explicit_env).expanduser())
    env_paths.append(REPO_ROOT / ".env")
    for root in _workspace_root_candidates():
        env_paths.append(root / ".env")
    seen_env_paths: set[str] = set()
    for dotenv_path in env_paths:
        key = str(dotenv_path)
        if key in seen_env_paths or not dotenv_path.exists():
            continue
        load_dotenv(dotenv_path, override=False)
        seen_env_paths.add(key)

    default_pooler = None
    default_encoder = None
    for root in [REPO_ROOT, *_workspace_root_candidates()]:
        candidate_pooler = root / "encoder_models" / "vjepa2_attention_poolers" / "ssv2-vitl-16x2x3.pt"
        candidate_encoder = root / "encoder_models" / "vitl.pt"
        if default_pooler is None and candidate_pooler.exists():
            default_pooler = candidate_pooler.resolve()
        if default_encoder is None and candidate_encoder.exists():
            default_encoder = candidate_encoder.resolve()

    defaults = {
        "DENSE_TEMPORAL_ENCODER_MODEL": "large",
        "DENSE_TEMPORAL_PREPROC_ID": "vjepa_rgb_256",
        "DENSE_TEMPORAL_INFERENCE_DTYPE": "bf16",
        "DENSE_TEMPORAL_DEVICE": "cuda",
        "DENSE_TEMPORAL_DECODE_DEVICE": "gpu",
        "DENSE_TEMPORAL_DECODE_GPU_ID": "0",
        "DENSE_TEMPORAL_BATCH_SIZE": "64",
    }
    if default_pooler is not None:
        defaults["DENSE_TEMPORAL_POOLER_PATH"] = str(default_pooler)
    if default_encoder is not None:
        defaults["DENSE_TEMPORAL_ENCODER_CHECKPOINT"] = str(default_encoder)
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    _PREPARE_ENV_LOADED = True


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if path.is_dir():
        shutil.rmtree(path)


def _clear_cache_outputs() -> None:
    for path in (RAW_FEATURES_DIR, POOLERS_DIR, SPLITS_DIR, INDEX_DIR):
        _remove_path(path)
    MANIFEST_PATH.unlink(missing_ok=True)


def _load_snapshot_payload(
    *,
    run_id: str,
    space_id: Optional[str],
    run_number: Optional[int],
    snapshot_path: Optional[Path],
) -> Tuple[dict, str, int]:
    if snapshot_path is not None:
        snapshot = _read_json(snapshot_path)
        if not isinstance(snapshot, dict):
            raise ValueError(f"Invalid snapshot JSON at {snapshot_path}")
        resolved_space_id = str(space_id or "").strip()
        resolved_run_number = int(run_number or 0)
        if not resolved_space_id or resolved_run_number <= 0:
            raise ValueError("--snapshot-path requires --space-id and --run-number")
        return snapshot, resolved_space_id, resolved_run_number

    _load_repo_env_once()
    from .supabase import EmbeddingDBClient

    db = EmbeddingDBClient()
    run_data = db.get_embedding_run(str(run_id))
    snapshot = run_data.get("config_snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError(f"Embedding run {run_id} has no valid config_snapshot")
    resolved_space_id = str(space_id or run_data.get("space_id") or "").strip()
    resolved_run_number = int(run_number or run_data.get("run_number") or 0)
    if not resolved_space_id:
        raise ValueError(f"Embedding run {run_id} missing space_id")
    if resolved_run_number <= 0:
        raise ValueError(f"Embedding run {run_id} missing run_number")
    return snapshot, resolved_space_id, resolved_run_number


def _normalize_cfg_paths(cfg: Any) -> Any:
    pooler_raw = _normalize_workspace_path(cfg.model.pooler_path)
    encoder_raw = _normalize_workspace_path(cfg.model.encoder_checkpoint)
    pooler_path = pooler_raw or Path(str(cfg.model.pooler_path)).expanduser()
    encoder_checkpoint = encoder_raw or Path(str(cfg.model.encoder_checkpoint)).expanduser()
    return replace(
        cfg,
        model=replace(
            cfg.model,
            pooler_path=str(pooler_path),
            encoder_checkpoint=str(encoder_checkpoint),
        ),
    )


def materialize_run_source(
    *,
    run_id: str,
    space_id: Optional[str],
    run_number: Optional[int],
    snapshot_path: Optional[Path],
    force_reextract: bool,
    camera_include_regex: Optional[str] = None,
    video_include_regex: Optional[str] = None,
    path_include_regex: Optional[str] = None,
) -> Path:
    _ensure_dirs(CACHE_DIR)
    materialized_root = MATERIALIZED_RUNS_DIR / str(run_id) / "dense_temporal"
    if not force_reextract:
        required = (
            materialized_root / "snapshot.json",
            materialized_root / "resolved_config.json",
            materialized_root / "run_summary.json",
            materialized_root / "features",
            materialized_root / "labels",
        )
        if all(path.exists() for path in required):
            LOGGER.info("Reusing materialized autoresearch source: %s", materialized_root)
            return materialized_root

    if force_reextract and materialized_root.exists():
        shutil.rmtree(materialized_root)
    materialized_root.mkdir(parents=True, exist_ok=True)

    snapshot, resolved_space_id, resolved_run_number = _load_snapshot_payload(
        run_id=str(run_id),
        space_id=space_id,
        run_number=run_number,
        snapshot_path=snapshot_path,
    )
    cfg = parse_dense_temporal_snapshot(
        run_id=str(run_id),
        space_id=str(resolved_space_id),
        run_number=int(resolved_run_number),
        snapshot=snapshot,
        env=dict(os.environ),
    )
    cfg = _normalize_cfg_paths(cfg)
    from .materialize import (
        _build_stream_specs,
        _extract_stream_features,
        _stage_model_assets,
    )

    (materialized_root / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    (materialized_root / "resolved_config.json").write_text(
        json.dumps(
            {
                "run_id": cfg.run_id,
                "space_id": cfg.space_id,
                "run_number": cfg.run_number,
                "dataset_mode": cfg.dataset_mode,
                "temporal_structure_mode": cfg.temporal_structure_mode,
                "temporal_targets": asdict(cfg.temporal_targets),
                "model": asdict(cfg.model),
                "train": asdict(cfg.train),
                "videos": [asdict(video) for video in cfg.videos],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    pooler_staged, pooler_sha, encoder_model, _pooler_map_path, encoder_ckpt = _stage_model_assets(
        cfg=cfg,
        work_dir=materialized_root,
    )
    stream_specs = _build_stream_specs(
        cfg=cfg,
        work_dir=materialized_root,
        camera_include_regex=camera_include_regex,
        video_include_regex=video_include_regex,
        path_include_regex=path_include_regex,
    )
    _extract_stream_features(
        cfg=cfg,
        stream_specs=stream_specs,
        pooler_path=Path(pooler_staged),
        pooler_sha=str(pooler_sha),
        encoder_checkpoint=Path(encoder_ckpt),
        force_reextract=bool(force_reextract),
    )

    summary = {
        "run_id": cfg.run_id,
        "space_id": cfg.space_id,
        "run_number": cfg.run_number,
        "pooler_sha": str(pooler_sha),
        "encoder_model": str(encoder_model),
        "encoder_checkpoint": str(Path(encoder_ckpt).resolve()),
        "streams": [
            {
                "name": spec.name,
                "video_id": spec.video_id,
                "camera_id": spec.camera_id,
                "features_npz": str(spec.features_npz),
                "labels_json": str(spec.labels_json),
                "source_video": str(spec.source_video),
                "roi": dict(spec.roi) if isinstance(spec.roi, dict) else spec.roi,
                "fps": spec.fps,
                "supervised_start_frame": spec.supervised_start_frame,
                "supervised_end_frame": spec.supervised_end_frame,
                "num_cycles": spec.num_cycles,
            }
            for spec in stream_specs
        ],
        "tcn_train_result": None,
        "probe_phase1_result": None,
        "cyclecl_result": None,
        "final_tcn_checkpoint": None,
        "final_probe_checkpoint": str(pooler_staged),
        "resume_pooler_path": str(pooler_staged),
        "materialized_only": True,
    }
    (materialized_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info(
        "Materialized dense-temporal source run=%s space=%s run_number=%s streams=%d root=%s",
        cfg.run_id,
        cfg.space_id,
        cfg.run_number,
        len(stream_specs),
        materialized_root,
    )
    return materialized_root


def _snapshot_materialized_run_id(*, snapshot_path: Path, space_id: Optional[str], run_number: Optional[int]) -> str:
    resolved_space_id = str(space_id or "").strip()
    resolved_run_number = int(run_number or 0)
    if not resolved_space_id or resolved_run_number <= 0:
        raise ValueError("--snapshot-path requires --space-id and --run-number")
    safe_space_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", resolved_space_id).strip("-") or "snapshot"
    digest = hashlib.sha1(snapshot_path.resolve().read_bytes()).hexdigest()[:12]
    return f"snapshot__{safe_space_id}__{resolved_run_number}__{digest}"


def resolve_requested_source_dirs(
    *,
    source_run_dirs: Sequence[str],
    run_id: Optional[str],
    space_id: Optional[str],
    run_number: Optional[int],
    snapshot_path: Optional[Path],
    force_materialize: bool,
    camera_include_regex: Optional[str] = None,
    video_include_regex: Optional[str] = None,
    path_include_regex: Optional[str] = None,
) -> List[str]:
    resolved = [str(Path(raw).expanduser().resolve()) for raw in source_run_dirs]
    materialize_target = str(run_id) if run_id else None
    if materialize_target is None and snapshot_path is not None:
        materialize_target = _snapshot_materialized_run_id(
            snapshot_path=snapshot_path,
            space_id=space_id,
            run_number=run_number,
        )
    if materialize_target:
        materialized = materialize_run_source(
            run_id=str(materialize_target),
            space_id=space_id,
            run_number=run_number,
            snapshot_path=snapshot_path,
            force_reextract=bool(force_materialize),
            camera_include_regex=camera_include_regex,
            video_include_regex=video_include_regex,
            path_include_regex=path_include_regex,
        )
        resolved.append(str(materialized))
    return resolved


def _matches_filter(value: str, pattern: Optional[re.Pattern[str]]) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(str(value)))


def _compile_regex(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if pattern is None or not str(pattern).strip():
        return None
    return re.compile(str(pattern), flags=re.IGNORECASE)


def _discover_source_dirs(
    source_run_dirs: Sequence[str],
    source_globs: Sequence[str],
) -> List[Path]:
    paths: List[Path] = []
    for raw in source_run_dirs:
        path = Path(raw).expanduser().resolve()
        if (path / "features").exists() and (path / "labels").exists():
            paths.append(path)
    for pattern in source_globs:
        for match in sorted(Path("/").glob(pattern.lstrip("/")) if pattern.startswith("/") else Path().glob(pattern)):
            if (match / "features").exists() and (match / "labels").exists():
                paths.append(match.resolve())
    unique: List[Path] = []
    seen: set[str] = set()
    for path in sorted(paths):
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def _load_feature_index(source_run_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted((source_run_dir / "features").glob("*__features.npz")):
        video_id = path.name.split("__features.npz")[0]
        out[video_id] = path.resolve()
    return out


def _frame_to_ms(frame: Optional[int], fps: float) -> Optional[int]:
    if frame is None:
        return None
    return int(round(float(frame) * 1000.0 / float(fps)))


def _load_feature_metadata(path: Path) -> dict:
    data = np.load(path)
    embeddings = data["embeddings"]
    timestamps = data["timestamps_ms"]
    return {
        "embedding_dim": int(embeddings.shape[1]),
        "num_windows": int(embeddings.shape[0]),
        "timestamps_ms": timestamps.astype(np.int64, copy=False),
        "pooler_sha": str(data["pooler_sha"].item()) if "pooler_sha" in data else "",
        "camera_id": str(data["camera_id"].item()) if "camera_id" in data else "",
        "video_id": str(data["video_id"].item()) if "video_id" in data else "",
        "fps": float(data["fps"].item()) if "fps" in data else 25.0,
    }


def _zip_member_array_meta(npz_path: Path, member_name: str) -> Tuple[Tuple[int, ...], np.dtype, int, bool]:
    with zipfile.ZipFile(npz_path) as zf:
        info = zf.getinfo(f"{member_name}.npy")
        if info.compress_type != 0:
            raise RuntimeError(f"{npz_path}:{member_name}.npy is compressed; cannot memmap safely.")
        with npz_path.open("rb") as handle:
            handle.seek(info.header_offset)
            local = handle.read(30)
            signature, *_rest, fname_len, extra_len = struct.unpack("<IHHHHHIIIHH", local)
            if signature != 0x04034B50:
                raise RuntimeError(f"Unexpected zip local header for {npz_path}:{member_name}.npy")
            npy_offset = info.header_offset + 30 + int(fname_len) + int(extra_len)
            handle.seek(npy_offset)
            version = np.lib.format.read_magic(handle)
            shape, fortran_order, dtype = np.lib.format._read_array_header(handle, version)  # type: ignore[attr-defined]
            data_offset = int(handle.tell())
    return tuple(int(v) for v in shape), np.dtype(dtype), data_offset, bool(fortran_order)


@lru_cache(maxsize=2048)
def _cached_zip_member_array_meta(npz_path_str: str, member_name: str) -> Tuple[Tuple[int, ...], np.dtype, int, bool]:
    return _zip_member_array_meta(Path(npz_path_str), member_name)


def memmap_npz_member(npz_path: Path, member_name: str) -> np.memmap:
    shape, dtype, offset, fortran_order = _cached_zip_member_array_meta(str(npz_path), member_name)
    order = "F" if fortran_order else "C"
    return np.memmap(npz_path, dtype=dtype, mode="r", offset=offset, shape=shape, order=order)


def _extract_cycles_ms(label_obj: dict, fps: float) -> List[EventPair]:
    pairs: List[EventPair] = []
    for item in label_obj.get("cycles", []) or []:
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        if start_ms is None or end_ms is None:
            start_ms = _frame_to_ms(item.get("start_frame"), fps)
            end_ms = _frame_to_ms(item.get("end_frame"), fps)
        if start_ms is None or end_ms is None:
            continue
        start_i = int(start_ms)
        end_i = int(end_ms)
        if end_i <= start_i:
            continue
        pairs.append(EventPair(start_ms=start_i, end_ms=end_i))
    pairs.sort(key=lambda item: (item.start_ms, item.end_ms))
    return pairs


def _resolve_supervised_bounds_ms(label_obj: dict, cycles: Sequence[EventPair], fps: float) -> Optional[Tuple[int, int]]:
    start_ms = label_obj.get("supervised_start_ms")
    end_ms = label_obj.get("supervised_end_ms")
    if start_ms is None:
        start_ms = _frame_to_ms(label_obj.get("supervised_start_frame"), fps)
    if end_ms is None:
        end_ms = _frame_to_ms(label_obj.get("supervised_end_frame"), fps)
    if start_ms is None and cycles:
        start_ms = int(min(item.start_ms for item in cycles))
    if end_ms is None and cycles:
        end_ms = int(max(item.end_ms for item in cycles))
    if start_ms is None or end_ms is None:
        return None
    start_i = int(start_ms)
    end_i = int(end_ms)
    if end_i <= start_i:
        return None
    return start_i, end_i


def _range_to_indices(
    timestamps_ms: np.ndarray,
    start_ms: int,
    end_ms: int,
) -> Optional[Tuple[int, int]]:
    start_idx = int(np.searchsorted(timestamps_ms, int(start_ms), side="left"))
    end_idx = int(np.searchsorted(timestamps_ms, int(end_ms), side="right") - 1)
    start_idx = max(0, min(start_idx, int(timestamps_ms.shape[0]) - 1))
    end_idx = max(0, min(end_idx, int(timestamps_ms.shape[0]) - 1))
    if end_idx < start_idx:
        return None
    return start_idx, end_idx


def _complete_cycles_within(
    cycles: Sequence[EventPair],
    start_ms: int,
    end_ms: int,
) -> List[EventPair]:
    return [
        item
        for item in cycles
        if int(item.start_ms) >= int(start_ms) and int(item.end_ms) <= int(end_ms)
    ]


def _materialize_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        current = link_path.resolve() if link_path.is_symlink() else link_path
        if current == target_path.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target_path)


def _segment_to_dict(record: SegmentRecord) -> dict:
    data = asdict(record)
    data["event_pairs_ms"] = [
        {"start_ms": int(start_ms), "end_ms": int(end_ms)}
        for start_ms, end_ms in record.event_pairs_ms
    ]
    return data


def _record_from_dict(obj: dict) -> SegmentRecord:
    return SegmentRecord(
        segment_id=str(obj["segment_id"]),
        split=str(obj["split"]),
        video_id=str(obj["video_id"]),
        camera_id=str(obj["camera_id"]),
        source_run_dir=str(obj["source_run_dir"]),
        feature_path=str(obj["feature_path"]),
        label_path=str(obj["label_path"]),
        pooler_checkpoint=str(obj["pooler_checkpoint"]),
        pooler_sha=str(obj.get("pooler_sha") or ""),
        embedding_dim=int(obj["embedding_dim"]),
        token_dim=int(obj["token_dim"]),
        tokens_per_window=int(obj["tokens_per_window"]),
        num_total_windows=int(obj["num_total_windows"]),
        fps=float(obj["fps"]),
        supervised_start_ms=int(obj["supervised_start_ms"]),
        supervised_end_ms=int(obj["supervised_end_ms"]),
        supervised_start_idx=int(obj["supervised_start_idx"]),
        supervised_end_idx=int(obj["supervised_end_idx"]),
        eval_start_ms=(int(obj["eval_start_ms"]) if obj.get("eval_start_ms") is not None else None),
        eval_end_ms=(int(obj["eval_end_ms"]) if obj.get("eval_end_ms") is not None else None),
        eval_start_idx=(int(obj["eval_start_idx"]) if obj.get("eval_start_idx") is not None else None),
        eval_end_idx=(int(obj["eval_end_idx"]) if obj.get("eval_end_idx") is not None else None),
        event_pairs_ms=tuple(
            (int(item["start_ms"]), int(item["end_ms"]))
            for item in obj.get("event_pairs_ms", [])
        ),
    )


def build_cache(
    *,
    source_run_dirs: Sequence[str],
    source_globs: Sequence[str],
    camera_include_regex: Optional[str],
    video_include_regex: Optional[str],
    path_include_regex: Optional[str],
    val_ratio: float,
    seed: int,
    force: bool,
) -> dict:
    if force and CACHE_DIR.exists():
        _clear_cache_outputs()

    _ensure_dirs(CACHE_DIR)

    camera_re = _compile_regex(camera_include_regex)
    video_re = _compile_regex(video_include_regex)
    path_re = _compile_regex(path_include_regex)

    effective_globs = list(source_globs)
    if not source_run_dirs and not effective_globs:
        effective_globs = list(DEFAULT_SOURCE_GLOBS)
    dirs = _discover_source_dirs(source_run_dirs, effective_globs)
    if not dirs:
        raise RuntimeError(
            "No dense_temporal source dirs found. "
            "Pass --source-run-dir or --source-glob with an extracted feature root."
        )

    per_video_pooler: Dict[str, Path] = {}
    per_video_feature: Dict[str, Path] = {}
    per_video_meta: Dict[str, dict] = {}
    segment_records: List[SegmentRecord] = []
    seen_segments: set[str] = set()

    for source_run_dir in dirs:
        feature_index = _load_feature_index(source_run_dir)
        resolved_cfg_path = source_run_dir / "resolved_config.json"
        resolved_cfg = _read_json(resolved_cfg_path) if resolved_cfg_path.exists() else {}
        model_cfg = resolved_cfg.get("model") or {}
        pooler_ckpt = _normalize_workspace_path(model_cfg.get("pooler_path"))
        snapshot_obj = _read_json(source_run_dir / "snapshot.json") if (source_run_dir / "snapshot.json").exists() else {}
        video_path_index = {
            str(item.get("video_id") or "").strip(): str(item.get("path") or "")
            for item in snapshot_obj.get("videos", [])
            if isinstance(item, dict)
        }
        labels_dir = source_run_dir / "labels"
        for label_path in sorted(labels_dir.glob("*.json")):
            label_obj = _read_json(label_path)
            video_id = str(label_obj.get("video_id") or label_path.name.split("__")[0]).strip()
            camera_id = str(label_obj.get("camera_id") or "").strip()
            feature_path = feature_index.get(video_id)
            if feature_path is None:
                continue
            feature_meta = per_video_meta.get(video_id)
            if feature_meta is None:
                feature_meta = _load_feature_metadata(feature_path)
                per_video_meta[video_id] = feature_meta
                per_video_feature[video_id] = feature_path
                if pooler_ckpt is not None:
                    per_video_pooler[video_id] = pooler_ckpt
            raw_path = str(video_path_index.get(video_id) or feature_path)
            if not _matches_filter(camera_id, camera_re):
                continue
            if not _matches_filter(video_id, video_re):
                continue
            if not _matches_filter(raw_path or str(feature_path), path_re):
                continue

            fps = float(label_obj.get("fps") or feature_meta["fps"] or 25.0)
            cycles = _extract_cycles_ms(label_obj, fps)
            supervised = _resolve_supervised_bounds_ms(label_obj, cycles, fps)
            if supervised is None:
                continue
            supervised_start_ms, supervised_end_ms = supervised
            timestamps_ms = feature_meta["timestamps_ms"]
            supervised_idx = _range_to_indices(timestamps_ms, supervised_start_ms, supervised_end_ms)
            if supervised_idx is None:
                continue
            supervised_start_idx, supervised_end_idx = supervised_idx
            complete_cycles = _complete_cycles_within(cycles, supervised_start_ms, supervised_end_ms)
            eval_start_ms: Optional[int] = None
            eval_end_ms: Optional[int] = None
            eval_start_idx: Optional[int] = None
            eval_end_idx: Optional[int] = None
            if complete_cycles:
                eval_start_ms = int(complete_cycles[0].start_ms)
                eval_end_ms = int(complete_cycles[-1].end_ms)
                eval_idx = _range_to_indices(timestamps_ms, eval_start_ms, eval_end_ms)
                if eval_idx is not None:
                    eval_start_idx, eval_end_idx = eval_idx

            split = "val" if _stable_fraction(video_id, seed) < float(val_ratio) else "train"
            segment_tag = label_path.stem.split("__", 1)[-1]
            segment_id = f"{video_id}__{segment_tag}"
            if segment_id in seen_segments:
                continue
            seen_segments.add(segment_id)

            token_shape, token_dtype, _token_offset, _token_fortran = _cached_zip_member_array_meta(
                str(feature_path),
                "tokens",
            )
            pooler_sha = str(feature_meta.get("pooler_sha") or "")
            record = SegmentRecord(
                segment_id=segment_id,
                split=split,
                video_id=video_id,
                camera_id=camera_id or str(feature_meta.get("camera_id") or ""),
                source_run_dir=str(source_run_dir),
                feature_path=str(feature_path),
                label_path=str(label_path),
                pooler_checkpoint=str(per_video_pooler.get(video_id) or ""),
                pooler_sha=pooler_sha,
                embedding_dim=int(feature_meta["embedding_dim"]),
                token_dim=int(token_shape[-1]),
                tokens_per_window=int(token_shape[1]),
                num_total_windows=int(feature_meta["num_windows"]),
                fps=fps,
                supervised_start_ms=int(supervised_start_ms),
                supervised_end_ms=int(supervised_end_ms),
                supervised_start_idx=int(supervised_start_idx),
                supervised_end_idx=int(supervised_end_idx),
                eval_start_ms=(int(eval_start_ms) if eval_start_ms is not None else None),
                eval_end_ms=(int(eval_end_ms) if eval_end_ms is not None else None),
                eval_start_idx=(int(eval_start_idx) if eval_start_idx is not None else None),
                eval_end_idx=(int(eval_end_idx) if eval_end_idx is not None else None),
                event_pairs_ms=tuple((int(item.start_ms), int(item.end_ms)) for item in complete_cycles),
            )
            segment_records.append(record)

    if not segment_records:
        raise RuntimeError("No matching segment records found after filtering.")

    for video_id, feature_path in per_video_feature.items():
        _materialize_symlink(RAW_FEATURES_DIR / f"{video_id}__features.npz", feature_path)
    for video_id, pooler_path in per_video_pooler.items():
        if pooler_path.exists():
            target = POOLERS_DIR / f"{video_id}__{pooler_path.name}"
            _materialize_symlink(target, pooler_path)

    train_records = sorted((item for item in segment_records if item.split == "train"), key=lambda item: item.segment_id)
    val_records = sorted((item for item in segment_records if item.split == "val"), key=lambda item: item.segment_id)
    val_eval_records = sorted(
        (
            item
            for item in val_records
            if item.eval_start_idx is not None and item.eval_end_idx is not None and item.event_pairs_ms
        ),
        key=lambda item: item.segment_id,
    )

    SPLITS_DIR.joinpath("train_videos.json").write_text(
        json.dumps(sorted({item.video_id for item in train_records}), indent=2),
        encoding="utf-8",
    )
    SPLITS_DIR.joinpath("val_videos.json").write_text(
        json.dumps(sorted({item.video_id for item in val_records}), indent=2),
        encoding="utf-8",
    )
    with TRAIN_SEGMENTS_PATH.open("w", encoding="utf-8") as handle:
        for item in train_records:
            handle.write(json.dumps(_segment_to_dict(item), default=_json_default) + "\n")
    with VAL_SEGMENTS_PATH.open("w", encoding="utf-8") as handle:
        for item in val_records:
            handle.write(json.dumps(_segment_to_dict(item), default=_json_default) + "\n")
    with VAL_EVAL_SEGMENTS_PATH.open("w", encoding="utf-8") as handle:
        for item in val_eval_records:
            handle.write(json.dumps(_segment_to_dict(item), default=_json_default) + "\n")

    summary = {
        "cache_version": CACHE_VERSION,
        "time_budget_seconds": TIME_BUDGET,
        "total_timeout_seconds": TOTAL_TIMEOUT_SECONDS,
        "pair_tolerance_ms": PAIR_TOLERANCE_MS,
        "val_ratio": float(val_ratio),
        "seed": int(seed),
        "repo_root": str(REPO_ROOT),
        "workspace_roots": [str(path) for path in _workspace_root_candidates()],
        "source_run_dirs": [str(path) for path in dirs],
        "train_videos": len({item.video_id for item in train_records}),
        "val_videos": len({item.video_id for item in val_records}),
        "train_segments": len(train_records),
        "val_segments": len(val_records),
        "val_eval_segments": len(val_eval_records),
        "decode_config": asdict(DEFAULT_DECODE_CONFIG),
    }
    MANIFEST_PATH.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return summary


def load_manifest(cache_root: Optional[Path] = None) -> dict:
    root = cache_root or CACHE_DIR
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Cache manifest missing: {path}. Run `python prepare.py` first.")
    return _read_json(path)


def load_split_records(
    split: Literal["train", "val", "val_eval"],
    *,
    cache_root: Optional[Path] = None,
) -> List[SegmentRecord]:
    root = cache_root or CACHE_DIR
    if split == "train":
        path = root / "index" / "train_segments.jsonl"
    elif split == "val":
        path = root / "index" / "val_segments.jsonl"
    elif split == "val_eval":
        path = root / "index" / "val_eval_segments.jsonl"
    else:
        raise ValueError(f"Unsupported split={split!r}")
    if not path.exists():
        raise FileNotFoundError(f"Missing split index: {path}. Run `python prepare.py` first.")
    out: List[SegmentRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            out.append(_record_from_dict(json.loads(line)))
    return out


def _slice_bounds(record: SegmentRecord, *, use_eval_span: bool) -> Tuple[int, int]:
    if use_eval_span and record.eval_start_idx is not None and record.eval_end_idx is not None:
        return int(record.eval_start_idx), int(record.eval_end_idx)
    return int(record.supervised_start_idx), int(record.supervised_end_idx)


def load_segment_arrays(
    record: SegmentRecord,
    *,
    representation: Literal["pooled_z0", "tokens", "both"] = "pooled_z0",
    use_eval_span: bool = False,
) -> dict:
    feature_path = Path(record.feature_path)
    lo, hi = _slice_bounds(record, use_eval_span=use_eval_span)
    hi = min(int(hi), int(record.num_total_windows) - 1)
    if hi < lo:
        raise ValueError(f"Invalid slice for {record.segment_id}: {lo}>{hi}")

    payload: dict = {
        "record": record,
        "slice_start_idx": int(lo),
        "slice_end_idx": int(hi),
    }
    data = np.load(feature_path)
    payload["timestamps_ms"] = data["timestamps_ms"][lo : hi + 1].astype(np.int64, copy=False)
    if representation in {"pooled_z0", "both"}:
        payload["pooled_z0"] = data["embeddings"][lo : hi + 1].astype(np.float32, copy=False)
    if representation in {"tokens", "both"}:
        tokens_mm = memmap_npz_member(feature_path, "tokens")
        payload["tokens"] = np.asarray(tokens_mm[lo : hi + 1], dtype=np.float16)
    return payload


def decode_event_pairs(
    probs: np.ndarray,
    timestamps_ms: np.ndarray,
    *,
    heads: Sequence[str] = ("start", "end", "cycle"),
) -> List[EventPair]:
    decoded = decode_start_end_pairs(
        probs=probs,
        timestamps_ms=timestamps_ms,
        heads=list(heads),
        cfg=DEFAULT_DECODE_CONFIG,
    )
    out: List[EventPair] = []
    for item in decoded.get("pairs", []):
        out.append(EventPair(start_ms=int(item["start_ms"]), end_ms=int(item["end_ms"])))
    return out


def _pair_cost_ms(pred: EventPair, gt: EventPair) -> int:
    return abs(int(pred.start_ms) - int(gt.start_ms)) + abs(int(pred.end_ms) - int(gt.end_ms))


def _pair_valid(pred: EventPair, gt: EventPair, tolerance_ms: int) -> bool:
    return (
        abs(int(pred.start_ms) - int(gt.start_ms)) <= int(tolerance_ms)
        and abs(int(pred.end_ms) - int(gt.end_ms)) <= int(tolerance_ms)
    )


def _pick_better_match(left: MatchResult, right: MatchResult) -> MatchResult:
    if left.matched_pairs != right.matched_pairs:
        return left if left.matched_pairs > right.matched_pairs else right
    if left.total_cost_ms != right.total_cost_ms:
        return left if left.total_cost_ms < right.total_cost_ms else right
    return left if left.assignment <= right.assignment else right


def best_pair_match(
    pred_pairs: Sequence[EventPair],
    gt_pairs: Sequence[EventPair],
    *,
    tolerance_ms: int = PAIR_TOLERANCE_MS,
) -> MatchResult:
    pred = tuple(sorted(pred_pairs, key=lambda item: (item.start_ms, item.end_ms)))
    gt = tuple(sorted(gt_pairs, key=lambda item: (item.start_ms, item.end_ms)))

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> MatchResult:
        if i >= len(pred) or j >= len(gt):
            return MatchResult(0, 0, ())
        best = _pick_better_match(solve(i + 1, j), solve(i, j + 1))
        if _pair_valid(pred[i], gt[j], tolerance_ms):
            tail = solve(i + 1, j + 1)
            candidate = MatchResult(
                matched_pairs=int(tail.matched_pairs + 1),
                total_cost_ms=int(tail.total_cost_ms + _pair_cost_ms(pred[i], gt[j])),
                assignment=((i, j),) + tail.assignment,
            )
            best = _pick_better_match(best, candidate)
        return best

    return solve(0, 0)


def evaluate_predictions(
    predictions_by_segment: Dict[str, Sequence[EventPair]],
    *,
    records: Optional[Sequence[SegmentRecord]] = None,
    cache_root: Optional[Path] = None,
) -> Dict[str, float]:
    eval_records = list(records) if records is not None else load_split_records("val_eval", cache_root=cache_root)
    tp = 0
    total_pred = 0
    total_gt = 0
    total_count_abs_error = 0.0
    total_start_abs_ms = 0
    total_end_abs_ms = 0
    total_matched = 0

    for record in eval_records:
        gt_pairs = [EventPair(start_ms=int(s), end_ms=int(e)) for s, e in record.event_pairs_ms]
        pred_pairs = list(predictions_by_segment.get(record.segment_id, ()))
        total_pred += len(pred_pairs)
        total_gt += len(gt_pairs)
        total_count_abs_error += abs(float(len(pred_pairs) - len(gt_pairs)))
        match = best_pair_match(pred_pairs, gt_pairs, tolerance_ms=PAIR_TOLERANCE_MS)
        tp += int(match.matched_pairs)
        total_matched += int(match.matched_pairs)
        for pred_idx, gt_idx in match.assignment:
            pred = pred_pairs[int(pred_idx)]
            gt = gt_pairs[int(gt_idx)]
            total_start_abs_ms += abs(int(pred.start_ms) - int(gt.start_ms))
            total_end_abs_ms += abs(int(pred.end_ms) - int(gt.end_ms))

    fp = total_pred - tp
    fn = total_gt - tp
    precision = tp / max(1.0, float(tp + fp))
    recall = tp / max(1.0, float(tp + fn))
    pair_f1 = 0.0 if (precision + recall) <= 0.0 else (2.0 * precision * recall / (precision + recall))
    count_mae = total_count_abs_error / max(1, len(eval_records))
    start_mae_ms = total_start_abs_ms / max(1, total_matched)
    end_mae_ms = total_end_abs_ms / max(1, total_matched)
    return {
        "val_pair_f1": float(pair_f1),
        "val_pair_precision": float(precision),
        "val_pair_recall": float(recall),
        "val_count_mae": float(count_mae),
        "val_start_mae_ms": float(start_mae_ms),
        "val_end_mae_ms": float(end_mae_ms),
        "matched_pairs": float(total_matched),
        "pred_pairs": float(total_pred),
        "gt_pairs": float(total_gt),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OneMed dense-temporal cache for autoresearch")
    parser.add_argument("--run-id", default=None, help="Canonical dense-temporal embedding run id to materialize from Supabase")
    parser.add_argument("--space-id", default=None, help="Optional explicit space id. Inferred from Supabase when omitted.")
    parser.add_argument("--run-number", type=int, default=None, help="Optional explicit run number. Inferred from Supabase when omitted.")
    parser.add_argument("--snapshot-path", default=None, help="Optional local snapshot JSON. Requires --space-id and --run-number.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache root override. The cache version folder is appended automatically.",
    )
    parser.add_argument(
        "--source-run-dir",
        action="append",
        default=[],
        help="Dense-temporal run root containing features/, labels/, resolved_config.json",
    )
    parser.add_argument(
        "--source-glob",
        action="append",
        default=[],
        help="Glob pattern for dense-temporal run roots. Defaults cover /tmp/embedding_runs and rnd_multiseed runs.",
    )
    parser.add_argument("--camera-include-regex", default=None, help="Optional camera_id regex filter")
    parser.add_argument("--video-include-regex", default=None, help="Optional video_id regex filter")
    parser.add_argument("--path-include-regex", default=None, help="Optional raw path / feature path regex filter")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Video-level validation ratio")
    parser.add_argument("--seed", type=int, default=42, help="Stable split seed")
    parser.add_argument("--force", action="store_true", help="Delete and rebuild the cache root")
    parser.add_argument(
        "--force-materialize",
        action="store_true",
        help="Force re-download / re-extract the run-backed dense-temporal source root before cache build.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cache_dir:
        configure_cache_paths(Path(args.cache_dir).expanduser().resolve() / CACHE_VERSION)
    source_run_dirs = resolve_requested_source_dirs(
        source_run_dirs=list(args.source_run_dir or []),
        run_id=(str(args.run_id).strip() if args.run_id else None),
        space_id=(str(args.space_id).strip() if args.space_id else None),
        run_number=(int(args.run_number) if args.run_number is not None else None),
        snapshot_path=(Path(args.snapshot_path).expanduser().resolve() if args.snapshot_path else None),
        force_materialize=bool(args.force_materialize),
        camera_include_regex=args.camera_include_regex,
        video_include_regex=args.video_include_regex,
        path_include_regex=args.path_include_regex,
    )
    summary = build_cache(
        source_run_dirs=source_run_dirs,
        source_globs=list(args.source_glob or []),
        camera_include_regex=args.camera_include_regex,
        video_include_regex=args.video_include_regex,
        path_include_regex=args.path_include_regex,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        force=bool(args.force),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
