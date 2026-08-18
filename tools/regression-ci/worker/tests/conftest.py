import sys
from pathlib import Path

_REGRESSION_CI = Path(__file__).resolve().parents[2]
if str(_REGRESSION_CI) not in sys.path:
    sys.path.insert(0, str(_REGRESSION_CI))
