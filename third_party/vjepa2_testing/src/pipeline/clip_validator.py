"""Validation utilities for downloaded clips."""

from __future__ import annotations

import logging
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Dict, List

import decord


def _strip_to_rgb(frame):
    if frame.ndim >= 3 and frame.shape[-1] > 3:
        return frame[..., :3]
    return frame


def count_frames_and_channels(path: Path, *, strip_alpha: bool = False) -> tuple[int, int | None]:
    """Return (num_frames, channel_count) for a video.

    Channel count is inferred from the first frame if available. When
    `strip_alpha` is True, extra channels are removed before counting so RGBA
    decodes that have been re-encoded still register as RGB.
    """
    try:
        vr = decord.VideoReader(str(path))
        num_frames = len(vr)
        channels = None
        if num_frames > 0:
            frame = vr[0].asnumpy()
            if strip_alpha:
                frame = _strip_to_rgb(frame)
            if frame.ndim >= 3:
                channels = int(frame.shape[-1])
        return num_frames, channels
    except Exception as exc:  # pragma: no cover - instrumentation path
        raise RuntimeError(f"Failed to inspect frames for {path}") from exc


def _reencode_to_rgb(path: Path, logger: logging.Logger) -> bool:
    """Convert a clip to 3-channel yuv420p in-place using ffmpeg."""

    with NamedTemporaryFile(prefix=path.stem + "_rgb_", suffix=path.suffix, dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        stderr_tail = (result.stderr or result.stdout or "").strip()
        logger.warning("RGB re-encode failed for %s: %s", path, stderr_tail[-400:])
        return False

    # Atomically replace the original clip.
    tmp_path.replace(path)
    return True


def validate_clips(
    class_to_dir: Dict[str, Path],
    expected_frames: int,
    logger: logging.Logger,
) -> Dict[str, List[Path]]:
    """Ensure clips contain the required number of frames."""

    validated: Dict[str, List[Path]] = {}
    for class_name, clip_dir in class_to_dir.items():
        if not clip_dir.exists():
            logger.warning("Clip directory missing for %s -> %s", class_name, clip_dir)
            validated[class_name] = []
            continue

        valid_paths: List[Path] = []
        discarded = 0
        non_rgb_failed = 0
        regenerated = 0
        for path in sorted(clip_dir.rglob("*.mp4")):
            try:
                frames, channels = count_frames_and_channels(path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", path, exc)
                discarded += 1
                continue

            if frames != expected_frames:
                logger.debug(
                    "Discarding %s: expected %d frames, found %d",
                    path,
                    expected_frames,
                    frames,
                )
                discarded += 1
                continue

            if channels is not None and channels != 3:
                logger.warning(
                    "Regenerating %s: expected RGB (3 channels), found %s",
                    path,
                    channels,
                )
                if not _reencode_to_rgb(path, logger):
                    discarded += 1
                    non_rgb_failed += 1
                    continue

                # Validate the regenerated clip twice to ensure it is clean RGB.
                ok = True
                for _ in range(2):
                    try:
                        post_frames, post_channels = count_frames_and_channels(path, strip_alpha=True)
                    except Exception as exc:
                        logger.warning("Post-conversion inspection failed for %s: %s", path, exc)
                        ok = False
                        break
                    if post_frames != expected_frames or post_channels != 3:
                        ok = False
                        break

                if not ok:
                    logger.warning("Discarding %s after failed RGB conversion/validation", path)
                    discarded += 1
                    non_rgb_failed += 1
                    continue

                regenerated += 1

            if channels is None:
                logger.debug("Discarding %s: unable to determine channel count", path)
                discarded += 1
                continue

            valid_paths.append(path)

        logger.info(
            "Validation summary for %s: %d valid clips, %d regenerated, %d discarded (non_rgb_failures=%d)",
            class_name,
            len(valid_paths),
            regenerated,
            discarded,
            non_rgb_failed,
        )
        validated[class_name] = valid_paths

    return validated
