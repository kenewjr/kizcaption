from pathlib import Path
import sys
# Source launcher only; installed package and executable need no path modification.
source = str(Path(__file__).resolve().parent / "src")
if source not in sys.path:
    sys.path.insert(0, source)
from lumacaption.main import main
if __name__ == "__main__":
    main()
