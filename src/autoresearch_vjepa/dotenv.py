"""
Minimal dotenv loader.

We intentionally avoid adding a dependency on python-dotenv. This loader supports
the subset of syntax we use in this repo (comments, optional `export`, optional
single/double quotes), and only populates missing environment variables by
default.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Set, Tuple


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_line(raw_line: str) -> Optional[Tuple[str, str]]:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].lstrip()

    if "=" not in line:
        return None

    key, value = line.split("=", 1)
    key = key.strip()
    if not key or not _KEY_RE.match(key):
        return None

    value = value.strip()
    if len(value) >= 2 and ((value[0] == value[-1] == '"') or (value[0] == value[-1] == "'")):
        value = value[1:-1]

    return key, value


def load_dotenv(path: Path, *, override: bool = False) -> Set[str]:
    """
    Populate `os.environ` from a dotenv-like file.

    - Supports leading `export` and quoted values.
    - Ignores blank lines and `#` comments.
    - By default only sets variables that are missing/empty (override=False).

    Returns the set of environment variable names that were set.
    """

    loaded: Set[str] = set()
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw_line)
        if not parsed:
            continue
        key, value = parsed

        if not override and os.environ.get(key):
            continue
        os.environ[key] = value
        loaded.add(key)

    return loaded

