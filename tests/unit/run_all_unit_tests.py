import pytest
import sys
from pathlib import Path

_UNIT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _UNIT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

pytest.main([str(_UNIT_DIR)])