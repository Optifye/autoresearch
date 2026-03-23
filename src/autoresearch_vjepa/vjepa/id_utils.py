"""Helpers for stable video/clip identifiers with camera disambiguation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TS_RE = re.compile(r"(\d{8}_\d{4})")


def video_id_from_path(path: Path, camera_hint: str | None = None) -> str:
    """Return a camera-scoped video_id derived from the path."""
    cam = camera_hint or (path.parent.name if path.parent else None)
    base = path.stem
    if cam:
        # Avoid double-prefixing if the stem is already scoped.
        if base.startswith(f"{cam}__"):
            return base
        return f"{cam}__{base}"
    return base


def clip_id_for(video_id: str, start_frame: int, end_frame: int) -> str:
    return f"{video_id}_f{start_frame:06d}-{end_frame:06d}"


def extract_ts_token(video_id: str) -> Optional[str]:
    """Best-effort extraction of the %Y%m%d_%H%M token from a video_id."""
    if "__" in video_id:
        tail = video_id.split("__", 1)[1]
        if TS_RE.fullmatch(tail):
            return tail
    if TS_RE.fullmatch(video_id):
        return video_id
    m = TS_RE.search(video_id)
    return m.group(1) if m else None


def parse_video_start_utc(video_id: str) -> datetime:
    ts = extract_ts_token(video_id)
    if not ts:
        raise ValueError(f"Cannot extract timestamp token from video_id={video_id}")
    return datetime.strptime(ts, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
