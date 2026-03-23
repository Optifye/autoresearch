"""Label parsing and mapping for Phase-1.5 interval model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import json
import numpy as np

from .feature_store import feature_sidecar_paths, open_feature_store


@dataclass
class EmbeddingStream:
    embeddings: np.ndarray  # [T, D]
    timestamps_ms: np.ndarray  # [T]
    clip_len: int
    stride: int
    frame_skip: int
    embedding_kind: str
    model_name: str
    pooler_sha: Optional[str]
    video_id: Optional[str] = None
    camera_id: Optional[str] = None
    fps: Optional[float] = None


@dataclass
class Interval:
    # end_ms is exclusive (interval is [start_ms, end_ms))
    start_ms: int
    end_ms: int


@dataclass
class MappedInterval:
    start_ms: int
    end_ms: int
    start_idx: int
    end_idx: int
    start_err_ms: int
    end_err_ms: int


@dataclass
class MappingStats:
    count: int
    mean_abs_start_err_ms: float
    mean_abs_end_err_ms: float
    max_abs_start_err_ms: int
    max_abs_end_err_ms: int


@dataclass
class IntervalLabels:
    video_id: str
    camera_id: str
    fps: Optional[float]
    intervals: List[Interval]


def _load_jsonl_embeddings(path: Path) -> EmbeddingStream:
    embeddings: List[List[float]] = []
    timestamps: List[int] = []
    clip_len: Optional[int] = None
    stride: Optional[int] = None
    frame_skip: Optional[int] = None
    embedding_kind: Optional[str] = None
    model_name: Optional[str] = None
    pooler_sha: Optional[str] = None
    video_id: Optional[str] = None
    camera_id: Optional[str] = None
    fps: Optional[float] = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            embedding = obj.get("embedding")
            if embedding is None and "embeddings_base" in obj:
                embedding = obj.get("embeddings_base")
            if embedding is None:
                raise ValueError("Missing embedding in JSONL record")
            embeddings.append(embedding)
            timestamps.append(int(obj.get("timestamp_ms") or obj.get("timestamp") or 0))
            if clip_len is None:
                clip_len = int(obj.get("clip_len") or 0)
            if stride is None:
                stride = int(obj.get("stride") or 0)
            if frame_skip is None:
                frame_skip = int(obj.get("frame_skip") or 0)
            if embedding_kind is None:
                embedding_kind = str(obj.get("embedding_kind") or obj.get("embeddings_kind") or "unknown")
            if model_name is None:
                model_name = str(obj.get("model_name") or "unknown")
            if pooler_sha is None:
                pooler_sha = obj.get("pooler_sha")
            if camera_id is None:
                camera_id = obj.get("camera_id")
            if video_id is None:
                video_id = obj.get("video_id") or obj.get("video")
            aux = obj.get("aux")
            if fps is None and isinstance(aux, dict):
                fps = aux.get("fps") or aux.get("fps_effective")

    if clip_len is None or stride is None or frame_skip is None:
        raise ValueError("Missing clip_len/stride/frame_skip in embeddings JSONL")
    if embedding_kind is None or model_name is None:
        raise ValueError("Missing embedding_kind/model_name in embeddings JSONL")

    emb_arr = np.asarray(embeddings, dtype=np.float32)
    ts_arr = np.asarray(timestamps, dtype=np.int64)

    return EmbeddingStream(
        embeddings=emb_arr,
        timestamps_ms=ts_arr,
        clip_len=clip_len,
        stride=stride,
        frame_skip=frame_skip,
        embedding_kind=embedding_kind,
        model_name=model_name,
        pooler_sha=pooler_sha,
        video_id=video_id,
        camera_id=camera_id,
        fps=float(fps) if fps is not None else None,
    )


def _load_npz_embeddings(path: Path) -> EmbeddingStream:
    path = Path(path).expanduser().resolve()
    use_feature_store = feature_sidecar_paths(path).root.exists()
    if not use_feature_store and path.exists():
        with np.load(path, allow_pickle=True) as probe:
            use_feature_store = "tokens" in set(probe.files)
    if use_feature_store:
        store = open_feature_store(path)
        return EmbeddingStream(
            embeddings=store.embeddings.astype(np.float32, copy=False),
            timestamps_ms=store.timestamps_ms.astype(np.int64, copy=False),
            clip_len=int(store.meta.clip_len),
            stride=int(store.meta.stride),
            frame_skip=int(store.meta.frame_skip),
            embedding_kind=str(store.meta.embedding_kind),
            model_name=str(store.meta.model_name),
            pooler_sha=store.meta.pooler_sha,
            video_id=store.meta.video_id,
            camera_id=store.meta.camera_id,
            fps=store.meta.fps,
        )

    data = np.load(path, allow_pickle=True)
    required = ["embeddings", "timestamps_ms", "clip_len", "stride", "frame_skip", "embedding_kind", "model_name"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"NPZ missing required keys: {missing}")
    embeddings = data["embeddings"]
    timestamps_ms = data["timestamps_ms"].astype(np.int64, copy=False)
    clip_len = int(data["clip_len"].item())
    stride = int(data["stride"].item())
    frame_skip = int(data["frame_skip"].item())
    embedding_kind = str(data["embedding_kind"].item())
    model_name = str(data["model_name"].item())
    pooler_sha = str(data["pooler_sha"].item()) if "pooler_sha" in data else None
    video_id = str(data["video_id"].item()) if "video_id" in data else None
    camera_id = str(data["camera_id"].item()) if "camera_id" in data else None
    fps = float(data["fps"].item()) if "fps" in data else None

    return EmbeddingStream(
        embeddings=embeddings.astype(np.float32, copy=False),
        timestamps_ms=timestamps_ms,
        clip_len=clip_len,
        stride=stride,
        frame_skip=frame_skip,
        embedding_kind=embedding_kind,
        model_name=model_name,
        pooler_sha=pooler_sha,
        video_id=video_id,
        camera_id=camera_id,
        fps=fps,
    )


def load_embeddings(path: Path) -> EmbeddingStream:
    if path.suffix.lower() == ".npz":
        stream = _load_npz_embeddings(path)
    else:
        stream = _load_jsonl_embeddings(path)
    validate_embeddings(stream)
    return stream


def validate_embeddings(stream: EmbeddingStream) -> None:
    if stream.embedding_kind != "base":
        raise ValueError(f"embedding_kind must be 'base' (got {stream.embedding_kind})")
    if stream.embeddings.ndim != 2:
        raise ValueError("embeddings must be [T, D]")
    if stream.timestamps_ms.ndim != 1 or stream.timestamps_ms.shape[0] != stream.embeddings.shape[0]:
        raise ValueError("timestamps length mismatch")
    if np.any(np.diff(stream.timestamps_ms) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if stream.clip_len <= 0 or stream.stride <= 0:
        raise ValueError("clip_len/stride must be positive")


def load_interval_labels(path: Path) -> IntervalLabels:
    data = json.loads(path.read_text(encoding="utf-8"))
    intervals = []
    fps = data.get("fps")
    for item in data.get("intervals", []):
        if "start_ms" in item and "end_ms" in item:
            start_ms = int(item["start_ms"])
            end_ms = int(item["end_ms"])
        else:
            if fps is None:
                raise ValueError("fps required when start_frame/end_frame are provided")
            start_ms = int(round(float(item["start_frame"]) * 1000.0 / float(fps)))
            # end_ms is exclusive: include end_frame by adding 1 frame
            end_ms = int(round((float(item["end_frame"]) + 1.0) * 1000.0 / float(fps)))
        if end_ms <= start_ms:
            continue
        intervals.append(Interval(start_ms=start_ms, end_ms=end_ms))
    return IntervalLabels(
        video_id=str(data.get("video_id") or "unknown"),
        camera_id=str(data.get("camera_id") or "unknown"),
        fps=float(fps) if fps is not None else None,
        intervals=intervals,
    )


def map_intervals_to_indices(
    intervals: List[Interval], timestamps_ms: np.ndarray
) -> Tuple[List[MappedInterval], MappingStats]:
    mapped: List[MappedInterval] = []
    start_errs = []
    end_errs = []
    for interval in intervals:
        start_ms = interval.start_ms
        end_ms = interval.end_ms
        start_idx = int(np.searchsorted(timestamps_ms, start_ms, side="left"))
        # end_idx is last timestamp strictly before end_ms (exclusive interval end)
        end_idx = int(np.searchsorted(timestamps_ms, end_ms, side="left") - 1)
        if start_idx < 0:
            start_idx = 0
        if end_idx >= len(timestamps_ms):
            end_idx = len(timestamps_ms) - 1
        if start_idx > end_idx:
            continue
        start_err = int(timestamps_ms[start_idx] - start_ms)
        end_err = int(end_ms - timestamps_ms[end_idx])
        mapped.append(
            MappedInterval(
                start_ms=start_ms,
                end_ms=end_ms,
                start_idx=start_idx,
                end_idx=end_idx,
                start_err_ms=start_err,
                end_err_ms=end_err,
            )
        )
        start_errs.append(abs(start_err))
        end_errs.append(abs(end_err))
    if mapped:
        stats = MappingStats(
            count=len(mapped),
            mean_abs_start_err_ms=float(np.mean(start_errs)),
            mean_abs_end_err_ms=float(np.mean(end_errs)),
            max_abs_start_err_ms=int(np.max(start_errs)),
            max_abs_end_err_ms=int(np.max(end_errs)),
        )
    else:
        stats = MappingStats(0, 0.0, 0.0, 0, 0)
    return mapped, stats


def build_dense_targets(timestamps_ms: np.ndarray, intervals: List[Interval]) -> np.ndarray:
    y = np.zeros(len(timestamps_ms), dtype=np.float32)
    if not intervals:
        return y
    for interval in intervals:
        mask = (timestamps_ms >= interval.start_ms) & (timestamps_ms < interval.end_ms)
        y[mask] = 1.0
    return y


def interval_durations_ms(intervals: List[Interval]) -> np.ndarray:
    if not intervals:
        return np.array([], dtype=np.float32)
    return np.array([max(0, i.end_ms - i.start_ms) for i in intervals], dtype=np.float32)
