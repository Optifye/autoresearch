"""Disk-backed feature storage helpers for dense temporal training."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import struct
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.lib.format import (
    _read_array_header,
    dtype_to_descr,
    open_memmap,
    read_magic,
    write_array_header_2_0,
)

LOGGER = logging.getLogger(__name__)

_LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
_LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class FeatureSidecarMeta:
    clip_len: int
    stride: int
    frame_skip: int
    embedding_kind: str
    model_name: str
    timestamp_alignment: Optional[str] = None
    pooler_sha: Optional[str] = None
    camera_id: Optional[str] = None
    video_id: Optional[str] = None
    fps: Optional[float] = None


@dataclass(frozen=True)
class FeatureSidecarPaths:
    root: Path
    tokens_npy: Path
    embeddings_npy: Path
    timestamps_npy: Path
    meta_json: Path


@dataclass(frozen=True)
class FeatureStore:
    npz_path: Path
    sidecar_paths: FeatureSidecarPaths
    tokens: np.ndarray
    embeddings: np.ndarray
    timestamps_ms: np.ndarray
    meta: FeatureSidecarMeta
    mode: str


def _scalar_from_npz(data: "np.lib.npyio.NpzFile", key: str) -> Optional[object]:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _as_optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def feature_sidecar_paths(npz_path: Path) -> FeatureSidecarPaths:
    npz_path = Path(npz_path).expanduser().resolve()
    root = npz_path.with_name(f"{npz_path.stem}.sidecar")
    return FeatureSidecarPaths(
        root=root,
        tokens_npy=root / "tokens.npy",
        embeddings_npy=root / "embeddings.npy",
        timestamps_npy=root / "timestamps_ms.npy",
        meta_json=root / "meta.json",
    )


def _load_sidecar_meta(paths: FeatureSidecarPaths) -> FeatureSidecarMeta:
    payload = json.loads(paths.meta_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid sidecar metadata at {paths.meta_json}")
    return FeatureSidecarMeta(
        clip_len=int(payload["clip_len"]),
        stride=int(payload["stride"]),
        frame_skip=int(payload["frame_skip"]),
        embedding_kind=str(payload["embedding_kind"]),
        model_name=str(payload["model_name"]),
        timestamp_alignment=_as_optional_str(payload.get("timestamp_alignment")),
        pooler_sha=_as_optional_str(payload.get("pooler_sha")),
        camera_id=_as_optional_str(payload.get("camera_id")),
        video_id=_as_optional_str(payload.get("video_id")),
        fps=_as_optional_float(payload.get("fps")),
    )


def _write_sidecar_meta(paths: FeatureSidecarPaths, meta: FeatureSidecarMeta) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.meta_json.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")


def read_feature_meta_from_npz(npz_path: Path) -> FeatureSidecarMeta:
    with np.load(npz_path, allow_pickle=True) as data:
        return FeatureSidecarMeta(
            clip_len=int(_scalar_from_npz(data, "clip_len") or 0),
            stride=int(_scalar_from_npz(data, "stride") or 0),
            frame_skip=int(_scalar_from_npz(data, "frame_skip") or 0),
            embedding_kind=str(_scalar_from_npz(data, "embedding_kind") or "base"),
            model_name=str(_scalar_from_npz(data, "model_name") or "unknown"),
            timestamp_alignment=_as_optional_str(_scalar_from_npz(data, "timestamp_alignment")),
            pooler_sha=_as_optional_str(_scalar_from_npz(data, "pooler_sha")),
            camera_id=_as_optional_str(_scalar_from_npz(data, "camera_id")),
            video_id=_as_optional_str(_scalar_from_npz(data, "video_id")),
            fps=_as_optional_float(_scalar_from_npz(data, "fps")),
        )


def _validate_feature_arrays(
    *,
    tokens: np.ndarray,
    embeddings: np.ndarray,
    timestamps_ms: np.ndarray,
    meta: FeatureSidecarMeta,
    source: Path,
) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"tokens must be [T,N,D] in {source} (got {tokens.shape})")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be [T,D] in {source} (got {embeddings.shape})")
    if timestamps_ms.ndim != 1:
        raise ValueError(f"timestamps_ms must be [T] in {source} (got {timestamps_ms.shape})")
    if int(tokens.shape[0]) != int(embeddings.shape[0]) or int(tokens.shape[0]) != int(timestamps_ms.shape[0]):
        raise ValueError(
            f"Feature length mismatch in {source}: tokens={tokens.shape} embeddings={embeddings.shape} timestamps={timestamps_ms.shape}"
        )
    if meta.clip_len <= 0 or meta.stride <= 0:
        raise ValueError(f"Invalid clip_len/stride in {source}: {meta.clip_len}/{meta.stride}")


def _load_sidecar_store(npz_path: Path, *, mode: str) -> FeatureStore:
    paths = feature_sidecar_paths(npz_path)
    meta = _load_sidecar_meta(paths)
    tokens = np.load(paths.tokens_npy, mmap_mode="r", allow_pickle=False)
    embeddings = np.load(paths.embeddings_npy, mmap_mode="r", allow_pickle=False)
    timestamps_ms = np.load(paths.timestamps_npy, mmap_mode="r", allow_pickle=False)
    _validate_feature_arrays(
        tokens=tokens,
        embeddings=embeddings,
        timestamps_ms=timestamps_ms,
        meta=meta,
        source=paths.root,
    )
    return FeatureStore(
        npz_path=Path(npz_path).expanduser().resolve(),
        sidecar_paths=paths,
        tokens=tokens,
        embeddings=embeddings,
        timestamps_ms=timestamps_ms,
        meta=meta,
        mode=str(mode),
    )


def feature_sidecars_ready(npz_path: Path) -> bool:
    try:
        _load_sidecar_store(npz_path, mode="sidecar_reuse")
        return True
    except Exception:
        return False


def feature_npz_ready(npz_path: Path) -> bool:
    npz_path = Path(npz_path).expanduser().resolve()
    if not npz_path.exists():
        return False
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            files = set(data.files)
        return {"tokens", "embeddings", "timestamps_ms"}.issubset(files)
    except Exception:
        return False


def _find_zip_member(zf: zipfile.ZipFile, member_name: str) -> str:
    for name in zf.namelist():
        if name.endswith(f"{member_name}.npy"):
            return name
    raise KeyError(f"Missing member {member_name}.npy")


def _get_zip_info(zf: zipfile.ZipFile, member_name: str) -> zipfile.ZipInfo:
    return zf.getinfo(_find_zip_member(zf, member_name))


def _zip_member_local_data_offset(npz_path: Path, info: zipfile.ZipInfo) -> int:
    with npz_path.open("rb") as handle:
        handle.seek(int(info.header_offset))
        header = handle.read(_LOCAL_FILE_HEADER.size)
    if len(header) != _LOCAL_FILE_HEADER.size:
        raise EOFError(f"Truncated local zip header in {npz_path} for {info.filename}")
    signature, *_unused, filename_len, extra_len = _LOCAL_FILE_HEADER.unpack(header)
    if int(signature) != int(_LOCAL_FILE_HEADER_SIGNATURE):
        raise ValueError(f"Invalid local zip header signature in {npz_path} for {info.filename}")
    return int(info.header_offset) + int(_LOCAL_FILE_HEADER.size) + int(filename_len) + int(extra_len)


def _memmap_npz_member(npz_path: Path, *, info: zipfile.ZipInfo) -> np.memmap:
    if int(info.compress_type) != int(zipfile.ZIP_STORED):
        raise ValueError(
            f"Feature member {info.filename} in {npz_path} is compressed (type={info.compress_type}); "
            "direct memmap requires ZIP_STORED"
        )
    npy_offset = _zip_member_local_data_offset(npz_path, info)
    with npz_path.open("rb") as handle:
        handle.seek(npy_offset)
        version = read_magic(handle)
        shape, fortran_order, dtype = _read_array_header(handle, version)
        array_offset = int(handle.tell())
    order = "F" if bool(fortran_order) else "C"
    return np.memmap(npz_path, mode="r", dtype=dtype, shape=shape, order=order, offset=array_offset)


def _load_npz_member_store(npz_path: Path, *, mode: str) -> FeatureStore:
    npz_path = Path(npz_path).expanduser().resolve()
    with zipfile.ZipFile(npz_path) as zf:
        tokens = _memmap_npz_member(npz_path, info=_get_zip_info(zf, "tokens"))
        embeddings = _memmap_npz_member(npz_path, info=_get_zip_info(zf, "embeddings"))
        timestamps_ms = _memmap_npz_member(npz_path, info=_get_zip_info(zf, "timestamps_ms"))
    meta = read_feature_meta_from_npz(npz_path)
    _validate_feature_arrays(
        tokens=tokens,
        embeddings=embeddings,
        timestamps_ms=timestamps_ms,
        meta=meta,
        source=npz_path,
    )
    return FeatureStore(
        npz_path=npz_path,
        sidecar_paths=feature_sidecar_paths(npz_path),
        tokens=tokens,
        embeddings=embeddings,
        timestamps_ms=timestamps_ms,
        meta=meta,
        mode=str(mode),
    )


def _stream_zip_member_to_npy(
    *,
    zf: zipfile.ZipFile,
    member_name: str,
    output_path: Path,
) -> None:
    member = _find_zip_member(zf, member_name)
    with zf.open(member) as src:
        version = read_magic(src)
        shape, fortran_order, dtype = _read_array_header(src, version)
        memmap = open_memmap(output_path, mode="w+", dtype=dtype, shape=shape, fortran_order=fortran_order)
        offset = int(memmap.offset)
        expected_nbytes = int(memmap.nbytes)
        memmap.flush()
        del memmap

        remaining = expected_nbytes
        with output_path.open("r+b") as dst:
            dst.seek(offset)
            while remaining > 0:
                chunk = src.read(min(remaining, 8 << 20))
                if not chunk:
                    break
                dst.write(chunk)
                remaining -= len(chunk)
        if remaining != 0:
            raise RuntimeError(f"Short read while copying {member_name} from zip member {member}")


def _replace_dir(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    src.rename(dest)


def _stage_sidecar_root(npz_path: Path) -> tuple[Path, FeatureSidecarPaths]:
    final_paths = feature_sidecar_paths(npz_path)
    stage_root = Path(
        tempfile.mkdtemp(prefix=f".{final_paths.root.name}.tmp.", dir=str(final_paths.root.parent))
    ).resolve()
    return stage_root, FeatureSidecarPaths(
        root=stage_root,
        tokens_npy=stage_root / "tokens.npy",
        embeddings_npy=stage_root / "embeddings.npy",
        timestamps_npy=stage_root / "timestamps_ms.npy",
        meta_json=stage_root / "meta.json",
    )


def ensure_feature_sidecars(npz_path: Path) -> str:
    npz_path = Path(npz_path).expanduser().resolve()
    try:
        _load_npz_member_store(npz_path, mode="npz_member_mmap")
        return "npz_member_mmap"
    except Exception:
        pass
    if feature_sidecars_ready(npz_path):
        return "sidecar_reuse"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing feature artifact: {npz_path}")

    meta = read_feature_meta_from_npz(npz_path)
    stage_root, stage_paths = _stage_sidecar_root(npz_path)
    try:
        with zipfile.ZipFile(npz_path) as zf:
            for name, dest in (
                ("tokens", stage_paths.tokens_npy),
                ("embeddings", stage_paths.embeddings_npy),
                ("timestamps_ms", stage_paths.timestamps_npy),
            ):
                _stream_zip_member_to_npy(zf=zf, member_name=name, output_path=dest)
        _write_sidecar_meta(stage_paths, meta)
        final_paths = feature_sidecar_paths(npz_path)
        _replace_dir(stage_root, final_paths.root)
    except Exception:
        shutil.rmtree(stage_root, ignore_errors=True)
        raise
    return "sidecar_backfill"


def open_feature_store(npz_path: Path) -> FeatureStore:
    npz_path = Path(npz_path).expanduser().resolve()
    try:
        return _load_npz_member_store(npz_path, mode="npz_member_mmap")
    except Exception:
        pass
    try:
        return _load_sidecar_store(npz_path, mode="sidecar_reuse")
    except Exception:
        mode = ensure_feature_sidecars(npz_path)
        if mode == "npz_member_mmap":
            return _load_npz_member_store(npz_path, mode=mode)
        return _load_sidecar_store(npz_path, mode=mode)


def rebuild_feature_npz_from_sidecar(npz_path: Path) -> Path:
    store = open_feature_store(npz_path)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{store.npz_path.name}.tmp.",
        suffix=".npz",
        dir=str(store.npz_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez(
            tmp_path,
            tokens=store.tokens,
            embeddings=store.embeddings,
            timestamps_ms=store.timestamps_ms,
            embedding_kind=str(store.meta.embedding_kind),
            model_name=str(store.meta.model_name),
            clip_len=int(store.meta.clip_len),
            stride=int(store.meta.stride),
            frame_skip=int(store.meta.frame_skip),
            timestamp_alignment=str(store.meta.timestamp_alignment or "window_center"),
            pooler_sha=str(store.meta.pooler_sha or ""),
            camera_id=str(store.meta.camera_id or ""),
            video_id=str(store.meta.video_id or ""),
            fps=float(store.meta.fps) if store.meta.fps is not None else np.nan,
        )
        tmp_path.replace(store.npz_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return store.npz_path


def _small_npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array, allow_pickle=False)
    return buf.getvalue()


def _copy_nbytes(*, src: io.BufferedReader, dst: io.BufferedWriter, count: int) -> None:
    remaining = int(count)
    while remaining > 0:
        chunk = src.read(min(remaining, 8 << 20))
        if not chunk:
            break
        dst.write(chunk)
        remaining -= len(chunk)
    if remaining != 0:
        raise RuntimeError(f"Short read while copying {count} bytes")


def _chunk_array_payload(path: Path) -> tuple[np.ndarray, int]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    offset = int(getattr(array, "offset", 0))
    if offset <= 0:
        raise RuntimeError(f"Expected memmap-backed chunk for {path}")
    return array, offset


def _write_chunked_array_member(
    *,
    zf: zipfile.ZipFile,
    member_name: str,
    chunk_paths: Sequence[Path],
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype,
) -> None:
    if not chunk_paths:
        raise ValueError(f"No chunk paths provided for {member_name}")

    header = {
        "descr": dtype_to_descr(np.dtype(expected_dtype)),
        "fortran_order": False,
        "shape": tuple(int(v) for v in expected_shape),
    }
    with zf.open(f"{member_name}.npy", mode="w", force_zip64=True) as member:
        write_array_header_2_0(member, header)
        expected_shape_tail = tuple(int(v) for v in expected_shape[1:])
        for chunk_path in chunk_paths:
            array, offset = _chunk_array_payload(Path(chunk_path))
            if array.dtype != np.dtype(expected_dtype):
                raise ValueError(
                    f"{member_name} chunk dtype mismatch for {chunk_path}: {array.dtype} != {np.dtype(expected_dtype)}"
                )
            if array.ndim != len(expected_shape):
                raise ValueError(
                    f"{member_name} chunk rank mismatch for {chunk_path}: {array.ndim} != {len(expected_shape)}"
                )
            if tuple(int(v) for v in array.shape[1:]) != expected_shape_tail:
                raise ValueError(
                    f"{member_name} chunk tail shape mismatch for {chunk_path}: {array.shape[1:]} != {expected_shape_tail}"
                )
            if not bool(array.flags.c_contiguous):
                raise ValueError(f"{member_name} chunk must be C-contiguous: {chunk_path}")
            with Path(chunk_path).open("rb") as src:
                src.seek(offset)
                _copy_nbytes(src=src, dst=member, count=int(array.nbytes))


def _write_feature_meta_members(zf: zipfile.ZipFile, *, meta: FeatureSidecarMeta) -> None:
    metadata_arrays = {
        "clip_len.npy": np.asarray(int(meta.clip_len), dtype=np.int64),
        "stride.npy": np.asarray(int(meta.stride), dtype=np.int64),
        "frame_skip.npy": np.asarray(int(meta.frame_skip), dtype=np.int64),
        "embedding_kind.npy": np.asarray(str(meta.embedding_kind)),
        "model_name.npy": np.asarray(str(meta.model_name)),
        "timestamp_alignment.npy": np.asarray(str(meta.timestamp_alignment or "window_center")),
        "pooler_sha.npy": np.asarray(str(meta.pooler_sha or "")),
        "camera_id.npy": np.asarray(str(meta.camera_id or "")),
        "video_id.npy": np.asarray(str(meta.video_id or "")),
        "fps.npy": np.asarray(float(meta.fps) if meta.fps is not None else np.nan, dtype=np.float64),
    }
    for member_name, array in metadata_arrays.items():
        zf.writestr(member_name, _small_npy_bytes(array))


def write_feature_npz_from_chunks(
    *,
    npz_path: Path,
    token_chunk_paths: Sequence[Path],
    embedding_chunk_paths: Sequence[Path],
    timestamp_chunk_paths: Sequence[Path],
    meta: FeatureSidecarMeta,
) -> FeatureStore:
    if not token_chunk_paths:
        raise ValueError("No token chunks provided")
    if len(token_chunk_paths) != len(embedding_chunk_paths) or len(token_chunk_paths) != len(timestamp_chunk_paths):
        raise ValueError("Chunk path lists must have the same length")

    token_shape_tail = None
    embed_dim = None
    total = 0
    chunk_lengths = []
    for token_path, embedding_path, ts_path in zip(token_chunk_paths, embedding_chunk_paths, timestamp_chunk_paths):
        token_chunk = np.load(token_path, mmap_mode="r", allow_pickle=False)
        embedding_chunk = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
        ts_chunk = np.load(ts_path, mmap_mode="r", allow_pickle=False)
        if token_shape_tail is None:
            token_shape_tail = tuple(int(v) for v in token_chunk.shape[1:])
        if embed_dim is None:
            embed_dim = int(embedding_chunk.shape[1])
        if tuple(int(v) for v in token_chunk.shape[1:]) != token_shape_tail:
            raise ValueError("Token chunk shape mismatch")
        if int(embedding_chunk.shape[1]) != embed_dim:
            raise ValueError("Embedding chunk dim mismatch")
        if int(token_chunk.shape[0]) != int(embedding_chunk.shape[0]) or int(token_chunk.shape[0]) != int(ts_chunk.shape[0]):
            raise ValueError("Chunk length mismatch")
        length = int(token_chunk.shape[0])
        if length <= 0:
            raise ValueError("Chunk length must be positive")
        total += length
        chunk_lengths.append(length)
    assert token_shape_tail is not None
    assert embed_dim is not None

    npz_path = Path(npz_path).expanduser().resolve()
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{npz_path.name}.tmp.",
        suffix=".npz",
        dir=str(npz_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            _write_chunked_array_member(
                zf=zf,
                member_name="tokens",
                chunk_paths=token_chunk_paths,
                expected_shape=(int(total), *token_shape_tail),
                expected_dtype=np.dtype(np.float16),
            )
            _write_chunked_array_member(
                zf=zf,
                member_name="embeddings",
                chunk_paths=embedding_chunk_paths,
                expected_shape=(int(total), int(embed_dim)),
                expected_dtype=np.dtype(np.float32),
            )
            _write_chunked_array_member(
                zf=zf,
                member_name="timestamps_ms",
                chunk_paths=timestamp_chunk_paths,
                expected_shape=(int(total),),
                expected_dtype=np.dtype(np.int64),
            )
            _write_feature_meta_members(zf, meta=meta)
        tmp_path.replace(npz_path)
        return _load_npz_member_store(npz_path, mode="npz_member_mmap")
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_feature_sidecars_from_chunks(
    *,
    npz_path: Path,
    token_chunk_paths: Sequence[Path],
    embedding_chunk_paths: Sequence[Path],
    timestamp_chunk_paths: Sequence[Path],
    meta: FeatureSidecarMeta,
) -> FeatureStore:
    return write_feature_npz_from_chunks(
        npz_path=npz_path,
        token_chunk_paths=token_chunk_paths,
        embedding_chunk_paths=embedding_chunk_paths,
        timestamp_chunk_paths=timestamp_chunk_paths,
        meta=meta,
    )
