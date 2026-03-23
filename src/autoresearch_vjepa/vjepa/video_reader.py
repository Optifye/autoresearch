"""FFmpeg-based video decoding helpers (NVDEC/CUVID preferred)."""

from __future__ import annotations

import json
import logging
import os
import selectors
import subprocess
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np

from .types import GPU_REQUIRED_SENTINEL

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMeta:
    fps: Optional[float]
    num_frames: Optional[int]
    width: Optional[int]
    height: Optional[int]
    backend: str


def _probe_metadata(path: Path) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int], Optional[str]]:
    # Use ffprobe to get fps, num_frames, width, height, codec.
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,avg_frame_rate,width,height,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        payload = json.loads(result.stdout)
        streams = payload.get("streams") if isinstance(payload, dict) else None
        if not streams or not isinstance(streams, list) or not streams:
            return None, None, None, None, None
        stream = streams[0] if isinstance(streams[0], dict) else None
        if not stream:
            return None, None, None, None, None
        codec = stream.get("codec_name")
        fps_raw = stream.get("avg_frame_rate")
        width = stream.get("width")
        height = stream.get("height")
        nb_frames_raw = stream.get("nb_frames")
        fps_val = None
        if isinstance(fps_raw, str) and fps_raw and "/" in fps_raw:
            num, denom = fps_raw.split("/", 1)
            try:
                denom_val = float(denom)
                if denom_val != 0:
                    fps_val = float(num) / denom_val
            except ValueError:
                fps_val = None
        elif isinstance(fps_raw, (int, float)):
            fps_val = float(fps_raw)
        elif isinstance(fps_raw, str) and fps_raw:
            try:
                fps_val = float(fps_raw)
            except ValueError:
                fps_val = None

        num_frames = None
        if isinstance(nb_frames_raw, int):
            num_frames = nb_frames_raw
        elif isinstance(nb_frames_raw, str) and nb_frames_raw.isdigit():
            num_frames = int(nb_frames_raw)

        width_val = int(width) if isinstance(width, (int, float, str)) and str(width).isdigit() else None
        height_val = int(height) if isinstance(height, (int, float, str)) and str(height).isdigit() else None
        codec_val = str(codec) if codec is not None else None
        return fps_val, num_frames, width_val, height_val, codec_val
    except Exception:
        return None, None, None, None, None


def _cuvid_decoder(codec: Optional[str]) -> Optional[str]:
    mapping = {
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "mpeg4": "mpeg4_cuvid",
        "mpeg2video": "mpeg2_cuvid",
        "mpeg1video": "mpeg1_cuvid",
        "vp8": "vp8_cuvid",
        "vp9": "vp9_cuvid",
        "av1": "av1_cuvid",
        "mjpeg": "mjpeg_cuvid",
        "vc1": "vc1_cuvid",
    }
    return mapping.get(codec or "")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _resolve_ffmpeg_pipe_timeouts() -> tuple[float, Optional[float]]:
    """Return (stall_timeout_s, total_timeout_s) for reading ffmpeg rawvideo stdout.

    - stall_timeout_s: maximum seconds without any stdout bytes before aborting.
    - total_timeout_s: maximum wall time for the decode (None disables).
    """
    stall_s = max(1.0, _env_float("CLUSTER_FFMPEG_PIPE_STALL_SEC", 120.0))
    total_s = _env_float("CLUSTER_FFMPEG_PIPE_TOTAL_SEC", 0.0)
    if total_s <= 0:
        return stall_s, None
    return stall_s, max(stall_s, total_s)


def _resolve_ffmpeg_read_chunk_bytes() -> int:
    return max(1 << 12, _env_int("CLUSTER_FFMPEG_PIPE_CHUNK_BYTES", 1 << 20))


def _resolve_ffmpeg_hw_init_retries() -> tuple[int, float]:
    retries = max(0, _env_int("CLUSTER_FFMPEG_HW_INIT_RETRIES", 2))
    sleep_s = max(0.0, _env_float("CLUSTER_FFMPEG_HW_INIT_RETRY_SLEEP_SEC", 1.0))
    return retries, sleep_s


def _is_retryable_ffmpeg_hw_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    tokens = (
        "cuda_error_no_device",
        "no device available for decoder",
        "no cuda-capable device is detected",
        "device creation failed",
    )
    return any(token in message for token in tokens)


def _set_nonblocking(fileobj) -> None:
    try:
        os.set_blocking(fileobj.fileno(), False)
    except Exception:
        return


def _iter_ffmpeg_rawvideo(
    *,
    proc: subprocess.Popen,
    width: int,
    height: int,
    step: int,
    max_frames: Optional[int],
    stall_timeout_s: float,
    total_timeout_s: Optional[float],
    chunk_bytes: int,
) -> Iterator[np.ndarray]:
    """Yield RGB frames from an ffmpeg rawvideo pipe with progress-based timeouts.

    The generator returns `True` (via StopIteration.value) when it stops because
    it reached max_frames from ffprobe metadata.
    """
    if proc.stdout is None:
        raise RuntimeError("ffmpeg stdout pipe is not available")
    frame_size = width * height * 3
    if frame_size <= 0:
        raise ValueError(f"Invalid frame_size={frame_size} for {width}x{height}")

    _set_nonblocking(proc.stdout)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    start = time.perf_counter()
    last_progress = start
    idx = 0
    stopped_by_max_frames = False
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                stopped_by_max_frames = True
                break

            buf = bytearray()
            while len(buf) < frame_size:
                now = time.perf_counter()
                if total_timeout_s is not None and (now - start) > total_timeout_s:
                    raise TimeoutError(f"ffmpeg rawvideo total timeout after {total_timeout_s:.1f}s")
                stall_for = now - last_progress
                if stall_for > stall_timeout_s:
                    raise TimeoutError(f"ffmpeg rawvideo stalled for {stall_for:.1f}s (limit {stall_timeout_s:.1f}s)")

                remaining_stall = max(0.0, stall_timeout_s - stall_for)
                timeout = min(1.0, remaining_stall) if remaining_stall > 0 else 0.0
                events = selector.select(timeout)
                if not events:
                    continue

                try:
                    chunk = os.read(proc.stdout.fileno(), min(chunk_bytes, frame_size - len(buf)))
                except BlockingIOError:
                    continue
                if not chunk:
                    break
                buf.extend(chunk)
                last_progress = time.perf_counter()

            if len(buf) < frame_size:
                break

            if idx % step == 0:
                raw = bytes(buf)
                arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                yield arr
            idx += 1
    finally:
        try:
            selector.unregister(proc.stdout)
        except Exception:
            pass
        selector.close()
        if stopped_by_max_frames:
            try:
                proc.stdout.close()
            except Exception:
                pass
    return stopped_by_max_frames


class VideoStream:
    def __init__(
        self,
        path: Path,
        *,
        prefer_decord: bool = True,
        target_fps: Optional[float] = None,
        frame_skip: int = 0,
        decode_device: str = "cpu",
        decode_gpu_id: int = 0,
        strict_gpu: bool = False,
        log_decodes: bool = True,
    ) -> None:
        self.path = Path(path)
        self.prefer_decord = prefer_decord
        self.target_fps = target_fps
        self.frame_skip = max(0, frame_skip)
        self._step = self.frame_skip + 1
        self._meta: Optional[VideoMeta] = None
        self._backend = None
        self._decode_device = decode_device
        self._decode_gpu_id = decode_gpu_id
        self._strict_gpu = bool(strict_gpu)
        self._log_decodes = bool(log_decodes)

        if self._strict_gpu and str(self._decode_device).strip().lower() != "gpu":
            raise ValueError(
                f"{GPU_REQUIRED_SENTINEL}: strict_gpu requires decode_device='gpu' (got {self._decode_device!r})"
            )

    def __iter__(self) -> Iterator[np.ndarray]:
        # Preferred path: decord (GPU if decode_device=gpu), then fall back to ffmpeg NVDEC, then CPU ffmpeg.
        if self.prefer_decord:
            try:
                # For GPU requests, only accept a real decord GPU path here. If decord
                # is CPU-only on this host, bubble up so ffmpeg NVDEC gets a chance.
                allow_cpu_fallback = self._decode_device.lower() != "gpu"
                yield from self._iter_decord(allow_cpu_fallback=allow_cpu_fallback)
                return
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Decord decode failed on %s (device=%s id=%s): %s; falling back to ffmpeg",
                    self.path,
                    self._decode_device,
                    self._decode_gpu_id,
                    exc,
                )
        if self._decode_device.lower() == "gpu":
            hw_retries, retry_sleep_s = _resolve_ffmpeg_hw_init_retries()
            for attempt in range(hw_retries + 1):
                try:
                    yield from self._iter_ffmpeg_hw()
                    return
                except Exception as exc:  # pylint: disable=broad-except
                    retryable = _is_retryable_ffmpeg_hw_error(exc)
                    if retryable and attempt < hw_retries:
                        LOGGER.warning(
                            "ffmpeg hw decode transient init failure on %s (attempt %d/%d): %s; retrying in %.1fs",
                            self.path,
                            attempt + 1,
                            hw_retries + 1,
                            exc,
                            retry_sleep_s,
                        )
                        if retry_sleep_s > 0:
                            time.sleep(retry_sleep_s)
                        continue
                    if self._strict_gpu:
                        raise RuntimeError(f"{GPU_REQUIRED_SENTINEL}: GPU decode failed for {self.path}") from exc
                    LOGGER.warning("ffmpeg hw decode failed on %s (%s); retrying on CPU", self.path, exc)
                    break
        yield from self._iter_ffmpeg_cpu()

    @property
    def metadata(self) -> VideoMeta:
        if self._meta is None:
            fps, num_frames, width, height, _codec = _probe_metadata(self.path)
            # Backend is populated during actual decode; use a distinct marker when probed eagerly.
            self._meta = VideoMeta(
                fps=fps,
                num_frames=num_frames,
                width=width,
                height=height,
                backend="ffprobe",
            )
        return self._meta  # type: ignore[return-value]

    def _iter_ffmpeg_hw(self) -> Iterator[np.ndarray]:
        # Use ffmpeg with cuvid decoder + cuda hwaccel to dump frames as rawvideo (RGB24).
        fps, num_frames, width, height, codec = _probe_metadata(self.path)
        decoder = _cuvid_decoder(codec)
        if decoder is None:
            raise RuntimeError(f"No cuvid decoder for codec {codec}")
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg binary not found")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "cuda",
            "-hwaccel_device",
            str(self._decode_gpu_id),
            "-c:v",
            decoder,
            "-i",
            str(self.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        if width is None or height is None:
            raise RuntimeError("Missing width/height from ffprobe for hw decode")
        if self._log_decodes:
            LOGGER.info(
                "Decoding %s via ffmpeg_cuvid (device=%s id=%s codec=%s frames=%s size=%sx%s step=%s)",
                self.path,
                self._decode_device,
                self._decode_gpu_id,
                codec,
                num_frames,
                width,
                height,
                self._step,
            )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: PLW1509
        stderr_tail: list[str] = []

        def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            for raw in iter(proc.stderr.readline, b""):
                try:
                    stderr_tail.append(raw.decode("utf-8", errors="replace"))
                    if len(stderr_tail) > 50:
                        del stderr_tail[:10]
                except Exception:
                    continue

        thread = threading.Thread(target=_drain_stderr, daemon=True)
        thread.start()
        self._meta = VideoMeta(fps=fps, num_frames=num_frames, width=width, height=height, backend="ffmpeg_cuvid")
        stall_s, total_s = _resolve_ffmpeg_pipe_timeouts()
        chunk_bytes = _resolve_ffmpeg_read_chunk_bytes()
        stopped_by_max_frames = False
        try:
            stopped_by_max_frames = yield from _iter_ffmpeg_rawvideo(
                proc=proc,
                width=width,
                height=height,
                step=self._step,
                max_frames=num_frames,
                stall_timeout_s=stall_s,
                total_timeout_s=total_s,
                chunk_bytes=chunk_bytes,
            )
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()
            thread.join(timeout=0.2)
        if proc.returncode not in (0, None) and not stopped_by_max_frames:
            tail = "".join(stderr_tail).strip()
            raise RuntimeError(f"ffmpeg hw decode failed with code {proc.returncode}: {tail[-2000:]}")

    def _iter_ffmpeg_cpu(self) -> Iterator[np.ndarray]:
        # CPU decode via ffmpeg to rawvideo (RGB24).
        fps, num_frames, width, height, _codec = _probe_metadata(self.path)
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg binary not found")
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(self.path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        if width is None or height is None:
            raise RuntimeError("Missing width/height from ffprobe for cpu decode")
        if self._log_decodes:
            LOGGER.info(
                "Decoding %s via ffmpeg_cpu (codec=%s frames=%s size=%sx%s step=%s)",
                self.path,
                _codec,
                num_frames,
                width,
                height,
                self._step,
            )
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: PLW1509
        stderr_tail: list[str] = []

        def _drain_stderr() -> None:
            if proc.stderr is None:
                return
            for raw in iter(proc.stderr.readline, b""):
                try:
                    stderr_tail.append(raw.decode("utf-8", errors="replace"))
                    if len(stderr_tail) > 50:
                        del stderr_tail[:10]
                except Exception:
                    continue

        thread = threading.Thread(target=_drain_stderr, daemon=True)
        thread.start()
        self._meta = VideoMeta(fps=fps, num_frames=num_frames, width=width, height=height, backend="ffmpeg_cpu")
        stall_s, total_s = _resolve_ffmpeg_pipe_timeouts()
        chunk_bytes = _resolve_ffmpeg_read_chunk_bytes()
        stopped_by_max_frames = False
        try:
            stopped_by_max_frames = yield from _iter_ffmpeg_rawvideo(
                proc=proc,
                width=width,
                height=height,
                step=self._step,
                max_frames=num_frames,
                stall_timeout_s=stall_s,
                total_timeout_s=total_s,
                chunk_bytes=chunk_bytes,
            )
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except Exception:
                pass
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc.wait()
            thread.join(timeout=0.2)
        if proc.returncode not in (0, None) and not stopped_by_max_frames:
            tail = "".join(stderr_tail).strip()
            raise RuntimeError(f"ffmpeg cpu decode failed with code {proc.returncode}: {tail[-2000:]}")

    def _iter_decord(self, *, allow_cpu_fallback: bool = True) -> Iterator[np.ndarray]:
        import decord

        # Keep retries minimal: try GPU once, then (optionally) CPU once, otherwise fall back to ffmpeg.
        import os

        # Allow a slightly higher EOF retry to avoid early abort on tiny clips.
        os.environ.setdefault("DECORD_EOF_RETRY_MAX", "50000")
        backend = "decord_cpu"
        vr = None
        # Attempt GPU first if requested.
        if self._decode_device.lower() == "gpu":
            try:
                ctx = decord.gpu(self._decode_gpu_id)
                vr = decord.VideoReader(str(self.path), ctx=ctx, num_threads=1)
                backend = "decord_gpu"
            except Exception as exc:
                if self._strict_gpu:
                    raise RuntimeError(f"Decord GPU decode failed on {self.path} (id={self._decode_gpu_id})") from exc
                if not allow_cpu_fallback:
                    raise RuntimeError(
                        f"Decord GPU decode failed on {self.path} (id={self._decode_gpu_id})"
                    ) from exc
                LOGGER.warning(
                    "Decord GPU decode failed on %s (id=%s): %s; retrying decord CPU",
                    self.path,
                    self._decode_gpu_id,
                    exc,
                )
                vr = None
        if vr is None:
            # CPU fallback
            vr = decord.VideoReader(str(self.path), num_threads=1)
            backend = "decord_cpu"
        fps = float(vr.get_avg_fps()) if vr.get_avg_fps() else None
        num_frames = len(vr)
        first = vr[0].asnumpy()
        height, width = first.shape[0], first.shape[1]
        self._meta = VideoMeta(fps=fps, num_frames=num_frames, width=width, height=height, backend=backend)
        if self._log_decodes:
            LOGGER.info(
                "Decoding %s via %s (device=%s id=%s) frames=%s size=%sx%s step=%s",
                self.path,
                backend,
                self._decode_device,
                self._decode_gpu_id,
                num_frames,
                width,
                height,
                self._step,
            )
        for idx in range(0, num_frames, self._step):
            if idx == 0:
                yield first
                continue
            frame = vr[idx].asnumpy()
            yield frame

    def _iter_av(self) -> Iterator[np.ndarray]:
        import av

        container = av.open(str(self.path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream and stream.average_rate else None
        width = stream.codec_context.width if stream and stream.codec_context else None
        height = stream.codec_context.height if stream and stream.codec_context else None
        num_frames = stream.frames if stream and stream.frames else None
        self._meta = VideoMeta(fps=fps, num_frames=num_frames, width=width, height=height, backend="pyav")
        for idx, frame in enumerate(container.decode(video=0)):
            if idx % self._step != 0:
                continue
            arr = frame.to_ndarray(format="rgb24")
            yield arr
