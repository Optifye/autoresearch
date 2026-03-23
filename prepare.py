from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from autoresearch_vjepa.cache_contract import *  # noqa: F401,F403
from autoresearch_vjepa.cache_contract import main


if __name__ == "__main__":
    raise SystemExit(main())
