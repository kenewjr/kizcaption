from __future__ import annotations

from pathlib import Path
import sys

# Ensure src and root are on sys.path for all tests
root = Path(__file__).resolve().parents[1]
src = root / "src"
pkg = src / "lumacaption"
for _p in (str(pkg), str(src), str(root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
