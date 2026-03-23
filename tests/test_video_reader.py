from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.vjepa.video_reader import VideoStream


def test_iter_prefers_ffmpeg_hw_after_decord_gpu_failure(monkeypatch, tmp_path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")

    calls: list[str] = []

    def fake_iter_decord(self, *, allow_cpu_fallback: bool = True):
        calls.append(f"decord:{allow_cpu_fallback}")
        raise RuntimeError("decord gpu unavailable")
        yield  # pragma: no cover

    def fake_iter_ffmpeg_hw(self):
        calls.append("ffmpeg_hw")
        yield np.zeros((2, 2, 3), dtype=np.uint8)

    def fake_iter_ffmpeg_cpu(self):
        calls.append("ffmpeg_cpu")
        yield np.ones((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(VideoStream, "_iter_decord", fake_iter_decord)
    monkeypatch.setattr(VideoStream, "_iter_ffmpeg_hw", fake_iter_ffmpeg_hw)
    monkeypatch.setattr(VideoStream, "_iter_ffmpeg_cpu", fake_iter_ffmpeg_cpu)

    stream = VideoStream(path, decode_device="gpu", prefer_decord=True, log_decodes=False)
    frames = list(stream)

    assert len(frames) == 1
    assert calls == ["decord:False", "ffmpeg_hw"]


def test_iter_decord_can_raise_without_cpu_fallback(monkeypatch, tmp_path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"")

    fake_decord = types.SimpleNamespace()

    class FakeVideoReader:
        def __init__(self, source: str, ctx=None, num_threads: int = 1) -> None:
            del source, num_threads
            if ctx is not None:
                raise RuntimeError("CUDA not enabled")

        def get_avg_fps(self) -> float:
            return 30.0

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            del index

            class Frame:
                @staticmethod
                def asnumpy() -> np.ndarray:
                    return np.zeros((2, 2, 3), dtype=np.uint8)

            return Frame()

    fake_decord.gpu = lambda gpu_id: ("gpu", gpu_id)
    fake_decord.VideoReader = FakeVideoReader
    monkeypatch.setitem(sys.modules, "decord", fake_decord)

    stream = VideoStream(path, decode_device="gpu", prefer_decord=True, log_decodes=False)

    try:
        list(stream._iter_decord(allow_cpu_fallback=False))
    except RuntimeError as exc:
        assert "Decord GPU decode failed" in str(exc)
    else:
        raise AssertionError("expected decord gpu failure to bubble up")
