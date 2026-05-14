import pytest
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config

growingnn.core.config.ENABLE_LOGGING = False

pytest.main(["."])