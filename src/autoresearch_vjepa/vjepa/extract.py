"""Video staging and V-JEPA feature extraction for standalone prepare.py."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse

import cv2  # type: ignore
import numpy as np
import torch

from ..feature_store import (
    FeatureSidecarMeta,
    feature_npz_ready,
    feature_sidecar_paths,
    open_feature_store,
    rebuild_feature_npz_from_sidecar,
    write_feature_npz_from_chunks,
)
from ..s3_videos import ArchivedS3ObjectError, R2VideoDownloader, S3VideoDownloader
from .preprocess import coerce_vjepa_preproc_id, preprocess_frames_batch
from .runtime import VJEPABackboneManager, VJEPABackboneRecord, VJEPARuntimeConfig
from .window_stream import iter_windows

LOGGER = logging.getLogger(__name__)


def _is_s3_uri(path: str) -> bool:
    return str(path or "").strip().lower().startswith("s3://")


def _is_r2_uri(path: str) -> bool:
    return str(path or "").strip().lower().startswith("r2://")


def _to_r2_path_from_s3(path: str) -> str:
    parsed = urlparse(path)
    return f"r2://{parsed.netloc}/{parsed.path.lstrip('/')}"


def stage_video(path_or_s3: str, *, downloads_dir: Path) -> Path:
    path_str = str(path_or_s3).strip()
    if not _is_s3_uri(path_str) and not _is_r2_uri(path_str):
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Missing video file: {path}")
        return path.resolve()

    parsed = urlparse(path_str)
    bucket_or_ns = parsed.netloc
    key = parsed.path.lstrip("/")
    if not bucket_or_ns or not key:
        raise ValueError(f"Invalid object uri: {path_str}")
    suffix = Path(key).suffix or ".mp4"
    stem = Path(key).stem or "video"
    short = hashlib.sha256(f"{bucket_or_ns}/{key}".encode("utf-8")).hexdigest()[:12]
    local = downloads_dir / f"{stem}__{short}{suffix}"
    if local.exists() and local.stat().st_size > 0:
        return local

    downloads_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = str(downloads_dir)
    if _is_r2_uri(path_str):
        downloader = R2VideoDownloader(cache_dir=cache_dir)
        resolved = Path(downloader.get_video_path(path_str))
        if resolved != local and resolved.exists():
            shutil.copy2(resolved, local)
        return local if local.exists() else resolved

    s3_downloader = S3VideoDownloader(
        cache_dir=cache_dir,
        region_name=os.getenv("AWS_REGION") or None,
    )
    try:
        resolved = Path(s3_downloader.get_video_path(path_str))
        if resolved != local and resolved.exists():
            shutil.copy2(resolved, local)
        return local if local.exists() else resolved
    except (ArchivedS3ObjectError, Exception) as exc:
        if str(os.getenv("S3_VIDEO_FALLBACK_TO_R2", "1")).strip().lower() in {"0", "false", "no", "off"}:
            raise
        r2_path = _to_r2_path_from_s3(path_str)
        r2_downloader = R2VideoDownloader(cache_dir=cache_dir)
        LOGGER.warning("S3 download failed for %s (%s); retrying via %s", path_str, exc, r2_path)
        resolved = Path(r2_downloader.get_video_path(r2_path))
        if resolved != local and resolved.exists():
            shutil.copy2(resolved, local)
        return local if local.exists() else resolved


def probe_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 0.0:
        raise RuntimeError(f"Invalid FPS={fps} for video={video_path}")
    return float(fps)


def window_center_ms(*, start_frame: int, end_frame: int, fps: float) -> int:
    center_frame = 0.5 * (float(start_frame) + float(end_frame))
    return int(round(center_frame * 1000.0 / float(fps)))


@dataclass(frozen=True)
class ExtractConfig:
    clip_len: int
    stride: int
    frame_skip: int
    batch_size: int
    preproc_id: str
    decode_device: str
    decode_gpu_id: int
    device: str
    inference_dtype: str


class VJEPAFeatureExtractor:
    def __init__(
        self,
        *,
        pooler_root: Path,
        pooler_map_path: Path,
        pooler_sha: str,
        encoder_checkpoint: Path,
        config: ExtractConfig,
    ) -> None:
        runtime_cfg = VJEPARuntimeConfig(
            device=str(config.device),
            inference_dtype=str(config.inference_dtype),
            pooler_root=Path(pooler_root).expanduser().resolve(),
            pooler_map_path=Path(pooler_map_path).expanduser().resolve(),
            vjepa_encoder_checkpoint=Path(encoder_checkpoint).expanduser().resolve(),
        )
        self.manager = VJEPABackboneManager(runtime_cfg)
        self.record: VJEPABackboneRecord = self.manager.get(str(pooler_sha), clip_len=int(config.clip_len))
        self.pooler_sha = str(pooler_sha)
        self.config = config
        self.resolved_preproc = coerce_vjepa_preproc_id(record=self.record, requested=str(config.preproc_id))
        self.amp_dtype = torch.bfloat16 if str(config.inference_dtype).lower() in {"bf16", "bfloat16"} else torch.float16

    def extract_to_npz(
        self,
        *,
        video_path: Path,
        camera_id: str,
        roi: Optional[Dict[str, float]],
        output_npz: Path,
        stop_after_ms: Optional[int],
        force: bool,
    ) -> Path:
        output_npz = Path(output_npz).expanduser().resolve()
        sidecar_root = feature_sidecar_paths(output_npz).root
        if not force and (output_npz.exists() or sidecar_root.exists()):
            try:
                store = open_feature_store(output_npz)
                if not feature_npz_ready(output_npz):
                    rebuild_feature_npz_from_sidecar(output_npz)
                LOGGER.info("Reusing cached features: %s (%s)", output_npz, store.mode)
                return output_npz
            except (OSError, EOFError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
                LOGGER.warning("Cached features %s are unreadable/incomplete (%s); regenerating", output_npz, exc)
                output_npz.unlink(missing_ok=True)
                shutil.rmtree(sidecar_root, ignore_errors=True)

        if force:
            output_npz.unlink(missing_ok=True)
            shutil.rmtree(sidecar_root, ignore_errors=True)

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        info, windows = iter_windows(
            video_path=video_path,
            camera_hint=str(camera_id),
            roi=roi,
            clip_len=int(self.config.clip_len),
            stride=int(self.config.stride),
            frame_skip=int(self.config.frame_skip),
            decode_device=str(self.config.decode_device),
            decode_gpu_id=int(self.config.decode_gpu_id),
            prefer_decord=True,
        )

        chunk_dir = Path(tempfile.mkdtemp(prefix=f".{output_npz.stem}.chunks.", dir=str(output_npz.parent))).resolve()
        token_chunk_paths: list[Path] = []
        embedding_chunk_paths: list[Path] = []
        ts_chunk_paths: list[Path] = []
        pending = []
        total = 0
        chunk_idx = 0
        stop_ms = int(stop_after_ms) if stop_after_ms is not None else None
        warned_multi_view = False

        autocast = torch.amp.autocast if self.record.device.type == "cuda" else contextlib.nullcontext

        def _flush() -> None:
            nonlocal pending, total, warned_multi_view, chunk_idx
            if not pending:
                return
            frames = np.asarray([window.frames for window in pending], dtype=np.uint8)
            inputs = preprocess_frames_batch(frames, preproc_id=str(self.resolved_preproc))
            with self.record.lock:
                device_inputs = inputs.to(self.record.device, non_blocking=True)
                with torch.no_grad():
                    with (
                        autocast(device_type="cuda", dtype=self.amp_dtype)
                        if self.record.device.type == "cuda"
                        else autocast()
                    ):
                        encoder_out = self.record.encoder([[device_inputs]], clip_indices=None)
                        views = list(encoder_out)
                        if not views:
                            raise RuntimeError("V-JEPA encoder returned zero views")
                        if len(views) > 1 and not warned_multi_view:
                            LOGGER.warning("Encoder returned %d views; using first for probe training.", len(views))
                            warned_multi_view = True
                        tokens = views[0]
                        embeddings = self.record.pooler(tokens).squeeze(1)

            token_chunk = tokens.detach().to(torch.float16).cpu().numpy().astype(np.float16, copy=False)
            embedding_chunk = embeddings.detach().to(torch.float32).cpu().numpy().astype(np.float32, copy=False)
            ts_chunk = np.asarray(
                [
                    window_center_ms(
                        start_frame=int(window.start_frame),
                        end_frame=int(window.end_frame),
                        fps=float(info.fps),
                    )
                    for window in pending
                ],
                dtype=np.int64,
            )
            token_path = chunk_dir / f"tokens_{chunk_idx:05d}.npy"
            embedding_path = chunk_dir / f"embeddings_{chunk_idx:05d}.npy"
            ts_path = chunk_dir / f"timestamps_{chunk_idx:05d}.npy"
            np.save(token_path, token_chunk, allow_pickle=False)
            np.save(embedding_path, embedding_chunk, allow_pickle=False)
            np.save(ts_path, ts_chunk, allow_pickle=False)
            token_chunk_paths.append(token_path)
            embedding_chunk_paths.append(embedding_path)
            ts_chunk_paths.append(ts_path)
            total += len(pending)
            chunk_idx += 1
            pending = []

        try:
            for window in windows:
                if stop_ms is not None:
                    center_ms = window_center_ms(
                        start_frame=int(window.start_frame),
                        end_frame=int(window.end_frame),
                        fps=float(info.fps),
                    )
                    if int(center_ms) > int(stop_ms):
                        break
                pending.append(window)
                if len(pending) >= int(self.config.batch_size):
                    _flush()
            _flush()

            if total <= 0:
                raise RuntimeError(f"No windows extracted for video={video_path}")

            store = write_feature_npz_from_chunks(
                npz_path=output_npz,
                token_chunk_paths=token_chunk_paths,
                embedding_chunk_paths=embedding_chunk_paths,
                timestamp_chunk_paths=ts_chunk_paths,
                meta=FeatureSidecarMeta(
                    clip_len=int(self.config.clip_len),
                    stride=int(self.config.stride),
                    frame_skip=int(self.config.frame_skip),
                    embedding_kind="base",
                    model_name=str(self.pooler_sha),
                    timestamp_alignment="window_center",
                    pooler_sha=str(self.pooler_sha),
                    camera_id=str(camera_id),
                    video_id=str(info.video_id),
                    fps=float(info.fps_effective),
                ),
            )
            if np.any(np.diff(store.timestamps_ms) <= 0):
                raise RuntimeError("Timestamps must be strictly increasing")
            return output_npz
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
