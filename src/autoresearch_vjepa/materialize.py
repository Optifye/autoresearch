"""Dense-temporal materialization helpers used by prepare.py."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .contracts import DenseTemporalRunConfig
from .label_conversion import DenseLabelShard, build_dense_label_shards

LOGGER = logging.getLogger(__name__)


def _is_state_dict(obj: object) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(key, str) for key in obj.keys()):
        return False
    return any(torch.is_tensor(value) for value in obj.values())


def _infer_encoder_model_from_state(state: Dict[str, torch.Tensor]) -> Optional[str]:
    seen_dims = set()
    for value in state.values():
        if not torch.is_tensor(value):
            continue
        for dim in value.shape:
            if isinstance(dim, int):
                seen_dims.add(dim)
    if 1408 in seen_dims:
        return "giant"
    if 1024 in seen_dims:
        return "large"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_packaged_pooler(
    pooler_path: Path,
    *,
    encoder_model_hint: Optional[str],
) -> Tuple[str, str]:
    payload = torch.load(pooler_path, map_location="cpu")
    if isinstance(payload, dict):
        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            pooler_sha = metadata.get("pooler_sha")
            encoder_model = metadata.get("encoder_model")
            if pooler_sha and encoder_model:
                return str(pooler_sha), str(encoder_model)

    pooler_sha = _sha256_file(pooler_path)
    pooler_state = None
    if isinstance(payload, dict) and _is_state_dict(payload.get("classifier")):
        pooler_state = payload["classifier"]
    elif isinstance(payload, dict) and _is_state_dict(payload.get("pooler_state")):
        pooler_state = payload["pooler_state"]
    elif isinstance(payload, dict) and isinstance(payload.get("classifiers"), list) and payload.get("classifiers"):
        first = payload["classifiers"][0]
        if _is_state_dict(first):
            pooler_state = first
    elif _is_state_dict(payload):
        pooler_state = payload
    if pooler_state is None:
        raise RuntimeError(
            "Unsupported pooler checkpoint format. Expected packaged pooler metadata or raw checkpoint with pooler state."
        )

    if any(isinstance(key, str) and key.startswith("module.") for key in pooler_state.keys()):
        pooler_state = {key[len("module.") :]: value for key, value in pooler_state.items() if isinstance(key, str)}

    if not any(isinstance(key, str) and key.startswith("pooler.") for key in pooler_state.keys()):
        pooler_state = {
            (
                key
                if (isinstance(key, str) and (key.startswith("linear.") or key.startswith("pooler.")))
                else f"pooler.{key}"
            ): value
            for key, value in pooler_state.items()
        }

    non_linear = [key for key in pooler_state if isinstance(key, str) and not key.startswith("linear.")]
    if not non_linear:
        raise RuntimeError(f"Pooler checkpoint looks like a linear head: {pooler_path}")

    encoder_model = (encoder_model_hint or "").strip().lower()
    if not encoder_model:
        encoder_model = _infer_encoder_model_from_state(pooler_state) or ""
    if encoder_model not in {"large", "giant"}:
        raise RuntimeError(
            f"Unable to infer encoder_model for pooler {pooler_path}. Provide temporal_model.encoder_model."
        )

    metadata = {
        "pooler_sha": pooler_sha,
        "encoder_model": encoder_model,
        "source": "autoresearch_auto_packaged",
    }
    torch.save({"pooler_state": pooler_state, "metadata": metadata}, pooler_path)
    LOGGER.info(
        "Auto-packaged pooler checkpoint %s (sha=%s encoder_model=%s)",
        pooler_path,
        pooler_sha,
        encoder_model,
    )
    return str(pooler_sha), str(encoder_model)


def _override_pooler_encoder_checkpoint(pooler_path: Path, *, encoder_checkpoint: str) -> None:
    payload = torch.load(pooler_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Pooler checkpoint {pooler_path} did not contain dict payload")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata["encoder_checkpoint"] = str(encoder_checkpoint)
    payload["metadata"] = metadata
    pooler_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f"{pooler_path.name}.",
        suffix=".tmp",
        dir=str(pooler_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, pooler_path)
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class _StreamSpec:
    name: str
    video_id: str
    camera_id: str
    features_npz: Path
    labels_json: Path
    source_video: Optional[Path]
    roi: Optional[Dict[str, float]]
    fps: float
    supervised_start_frame: int
    supervised_end_frame: int
    num_cycles: int


def _write_labels_file(path: Path, *, shard: DenseLabelShard, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(shard.to_cycle_labels_json(fps=float(fps)), indent=2),
        encoding="utf-8",
    )


def _compile_regex(pattern: Optional[str]) -> Optional[re.Pattern[str]]:
    if pattern is None or not str(pattern).strip():
        return None
    return re.compile(str(pattern), flags=re.IGNORECASE)


def _matches_filter(value: str, pattern: Optional[re.Pattern[str]]) -> bool:
    if pattern is None:
        return True
    return bool(pattern.search(str(value or "")))


def _materialize_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        current = link_path.resolve() if link_path.is_symlink() else link_path
        if current == target_path.resolve():
            return
        link_path.unlink()
    link_path.symlink_to(target_path)


def _candidate_reuse_source_dirs(*, cfg: DenseTemporalRunConfig, work_dir: Path) -> List[Path]:
    if str(os.getenv("AUTORESEARCH_DISABLE_FEATURE_REUSE", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return []

    candidates: List[Path] = []
    raw_env = str(os.getenv("AUTORESEARCH_REUSE_SOURCE_RUN_DIRS", "")).strip()
    if raw_env:
        for raw in raw_env.split(os.pathsep):
            text = str(raw).strip()
            if text:
                candidates.append(Path(text).expanduser())

    legacy_root = Path("/tmp/autoresearch_minda_sources")
    if legacy_root.exists():
        candidates.extend(sorted(legacy_root.glob("*/dense_temporal")))

    seen: set[str] = set()
    compatible: List[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        key = str(candidate)
        if key in seen or candidate == work_dir:
            continue
        seen.add(key)
        resolved_cfg_path = candidate / "resolved_config.json"
        features_dir = candidate / "features"
        if not resolved_cfg_path.exists() or not features_dir.exists():
            continue
        try:
            resolved_cfg = json.loads(resolved_cfg_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_cfg = resolved_cfg.get("model") if isinstance(resolved_cfg.get("model"), dict) else {}
        train_cfg = resolved_cfg.get("train") if isinstance(resolved_cfg.get("train"), dict) else {}
        if str(resolved_cfg.get("space_id") or "").strip() != str(cfg.space_id):
            continue
        clip_len = train_cfg.get("clip_len")
        stride = train_cfg.get("stride")
        frame_skip = train_cfg.get("frame_skip")
        if int(-1 if clip_len is None else clip_len) != int(cfg.train.clip_len):
            continue
        if int(-1 if stride is None else stride) != int(cfg.train.stride):
            continue
        if int(-1 if frame_skip is None else frame_skip) != int(cfg.train.frame_skip):
            continue
        if str(model_cfg.get("encoder_model") or "").strip().lower() != str(cfg.model.encoder_model).strip().lower():
            continue
        if str(model_cfg.get("preproc_id") or "").strip() != str(cfg.model.preproc_id):
            continue
        compatible.append(candidate)
    return compatible


def _load_feature_meta(feature_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with np.load(feature_path) as data:
            return {
                "video_id": str(data["video_id"].item()) if "video_id" in data else "",
                "camera_id": str(data["camera_id"].item()) if "camera_id" in data else "",
                "fps": float(data["fps"].item()) if "fps" in data else 0.0,
                "clip_len": int(data["clip_len"].item()) if "clip_len" in data else -1,
                "stride": int(data["stride"].item()) if "stride" in data else -1,
                "frame_skip": int(data["frame_skip"].item()) if "frame_skip" in data else -1,
                "pooler_sha": str(data["pooler_sha"].item()) if "pooler_sha" in data else "",
            }
    except Exception:
        return None


def _discover_reusable_feature_paths(
    *,
    cfg: DenseTemporalRunConfig,
    work_dir: Path,
    pooler_sha: str,
) -> Dict[str, Path]:
    expected_videos = {
        str(video.video_id): str(video.camera_id)
        for video in cfg.videos
    }
    reusable: Dict[str, Path] = {}
    for source_dir in _candidate_reuse_source_dirs(cfg=cfg, work_dir=work_dir):
        for feature_path in sorted((source_dir / "features").glob("*__features.npz")):
            meta = _load_feature_meta(feature_path)
            if not meta:
                continue
            video_id = str(feature_path.name).split("__features.npz")[0].strip()
            if not video_id or video_id in reusable or video_id not in expected_videos:
                continue
            if str(meta.get("camera_id") or "").strip() != expected_videos[video_id]:
                continue
            clip_len = meta.get("clip_len")
            stride = meta.get("stride")
            frame_skip = meta.get("frame_skip")
            if int(-1 if clip_len is None else clip_len) != int(cfg.train.clip_len):
                continue
            if int(-1 if stride is None else stride) != int(cfg.train.stride):
                continue
            if int(-1 if frame_skip is None else frame_skip) != int(cfg.train.frame_skip):
                continue
            if str(meta.get("pooler_sha") or "").strip() != str(pooler_sha):
                continue
            reusable[video_id] = feature_path.resolve()
    return reusable


def _stage_model_assets(
    *,
    cfg: DenseTemporalRunConfig,
    work_dir: Path,
) -> Tuple[Path, str, str, Path, Path]:
    staged_dir = work_dir / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)

    pooler_src = Path(cfg.model.pooler_path).expanduser()
    if not pooler_src.exists():
        raise FileNotFoundError(f"Pooler checkpoint not found: {pooler_src}")
    encoder_ckpt = Path(cfg.model.encoder_checkpoint).expanduser()
    if not encoder_ckpt.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_ckpt}")

    pooler_staged = staged_dir / "pooler_pretrained.pt"
    shutil.copy2(pooler_src, pooler_staged)
    pooler_sha, encoder_model = _ensure_packaged_pooler(
        pooler_staged,
        encoder_model_hint=str(cfg.model.encoder_model),
    )
    _override_pooler_encoder_checkpoint(pooler_staged, encoder_checkpoint=str(encoder_ckpt.resolve()))
    pooler_map_path = staged_dir / "pooler_map.json"
    pooler_map_path.write_text(json.dumps({pooler_sha: str(pooler_staged)}, indent=2), encoding="utf-8")
    return pooler_staged, pooler_sha, str(encoder_model), pooler_map_path, encoder_ckpt


def _build_stream_specs(
    *,
    cfg: DenseTemporalRunConfig,
    work_dir: Path,
    reusable_feature_paths: Optional[Dict[str, Path]] = None,
    camera_include_regex: Optional[str] = None,
    video_include_regex: Optional[str] = None,
    path_include_regex: Optional[str] = None,
) -> List[_StreamSpec]:
    from .vjepa.extract import probe_video_fps, stage_video

    camera_re = _compile_regex(camera_include_regex)
    video_re = _compile_regex(video_include_regex)
    path_re = _compile_regex(path_include_regex)

    downloads_dir = work_dir / "downloads"
    features_dir = work_dir / "features"
    labels_dir = work_dir / "labels"
    for path in (downloads_dir, features_dir, labels_dir):
        path.mkdir(parents=True, exist_ok=True)

    candidate_videos = [video for video in cfg.videos if not video.is_for_test]
    if not candidate_videos:
        candidate_videos = list(cfg.videos)
    stream_specs: List[_StreamSpec] = []
    reusable_feature_paths = dict(reusable_feature_paths or {})

    for video in candidate_videos:
        if not _matches_filter(video.camera_id, camera_re):
            continue
        if not _matches_filter(video.video_id, video_re):
            continue
        if not _matches_filter(video.path, path_re):
            continue

        local_video: Optional[Path] = None
        if str(video.video_id) not in reusable_feature_paths:
            local_video = stage_video(video.path, downloads_dir=downloads_dir)
        if video.fps is not None:
            fps = float(video.fps)
        elif local_video is not None:
            fps = float(probe_video_fps(local_video))
        else:
            reusable_meta = _load_feature_meta(reusable_feature_paths[str(video.video_id)])
            reusable_fps = float((reusable_meta or {}).get("fps") or 0.0)
            if reusable_fps <= 0.0:
                raise RuntimeError(f"Missing fps for reused video without local source probe: {video.video_id}")
            fps = reusable_fps
        shards = build_dense_label_shards(
            video,
            label_map=dict(cfg.temporal_targets.label_map),
            action_labels=list(cfg.temporal_targets.action_labels),
            ignore_label=int(cfg.temporal_targets.ignore_label),
            fallback_idle_label=int(cfg.temporal_targets.label_map.get("idle", 0)),
            temporal_structure_mode=str(cfg.temporal_structure_mode),
        )
        if not shards:
            LOGGER.warning("Skipping video=%s: no labeled action cycles after dense conversion", video.video_id)
            continue

        feature_npz = features_dir / f"{video.video_id}__features.npz"
        for shard in shards:
            labels_json = labels_dir / f"{shard.shard_id}.json"
            _write_labels_file(labels_json, shard=shard, fps=float(fps))
            stream_specs.append(
                _StreamSpec(
                    name=str(shard.shard_id),
                    video_id=str(video.video_id),
                    camera_id=str(video.camera_id),
                    features_npz=feature_npz,
                    labels_json=labels_json,
                    source_video=local_video,
                    roi=video.roi,
                    fps=float(fps),
                    supervised_start_frame=int(shard.supervised_start_frame),
                    supervised_end_frame=int(shard.supervised_end_frame),
                    num_cycles=len(shard.cycles),
                )
            )

    if not stream_specs:
        raise RuntimeError("No dense-temporal training streams were produced from snapshot videos")
    return stream_specs


def _extract_stream_features(
    *,
    cfg: DenseTemporalRunConfig,
    stream_specs: Sequence[_StreamSpec],
    pooler_path: Path,
    pooler_sha: str,
    encoder_checkpoint: Path,
    force_reextract: bool,
    reusable_feature_paths: Optional[Dict[str, Path]] = None,
) -> None:
    from .vjepa.extract import ExtractConfig, VJEPAFeatureExtractor

    pooler_map_path = pooler_path.parent / "pooler_map.json"
    pooler_map_path.write_text(json.dumps({str(pooler_sha): str(pooler_path)}, indent=2), encoding="utf-8")
    extractor = VJEPAFeatureExtractor(
        pooler_root=pooler_path.parent,
        pooler_map_path=pooler_map_path,
        pooler_sha=str(pooler_sha),
        encoder_checkpoint=Path(encoder_checkpoint).expanduser(),
        config=ExtractConfig(
            clip_len=int(cfg.train.clip_len),
            stride=int(cfg.train.stride),
            frame_skip=int(cfg.train.frame_skip),
            batch_size=int(cfg.model.batch_size),
            preproc_id=str(cfg.model.preproc_id),
            decode_device=str(cfg.model.decode_device),
            decode_gpu_id=int(cfg.model.decode_gpu_id),
            device=str(cfg.model.device),
            inference_dtype=str(cfg.model.inference_dtype),
        ),
    )

    reusable_feature_paths = dict(reusable_feature_paths or {})
    reused_outputs: set[str] = set()
    extracted_count = 0
    jobs: Dict[str, Dict[str, Any]] = {}
    for spec in stream_specs:
        reusable = reusable_feature_paths.get(str(spec.video_id))
        if reusable is not None:
            _materialize_symlink(Path(spec.features_npz), reusable)
            reused_outputs.add(str(spec.features_npz))
            continue
        stop_after_ms = int(round(float(spec.supervised_end_frame) * 1000.0 / float(spec.fps)))
        key = str(spec.features_npz)
        current = jobs.get(key)
        if current is None:
            if spec.source_video is None:
                raise RuntimeError(f"Missing source_video for non-reused stream {spec.name}")
            jobs[key] = {
                "video_path": Path(spec.source_video),
                "camera_id": str(spec.camera_id),
                "roi": dict(spec.roi) if isinstance(spec.roi, dict) else spec.roi,
                "output_npz": Path(spec.features_npz),
                "stop_after_ms": int(stop_after_ms),
            }
            continue
        current["stop_after_ms"] = max(int(current["stop_after_ms"]), int(stop_after_ms))

    for job in jobs.values():
        extractor.extract_to_npz(
            video_path=Path(job["video_path"]),
            camera_id=str(job["camera_id"]),
            roi=job["roi"],
            output_npz=Path(job["output_npz"]),
            stop_after_ms=int(job["stop_after_ms"]),
            force=bool(force_reextract),
        )
        extracted_count += 1

    LOGGER.info(
        "Feature materialization complete: reused=%d extracted=%d total_streams=%d",
        int(len(reused_outputs)),
        int(extracted_count),
        int(len(stream_specs)),
    )
