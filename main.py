from __future__ import annotations

from pathlib import Path
import sys

# Ensure src and lumacaption packages are on sys.path
root = Path(__file__).resolve().parent
src = root / "src"
pkg = src / "lumacaption"
for _p in (str(root), str(src), str(pkg)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lumacaption.main import main

if __name__ == "__main__":
    main()
