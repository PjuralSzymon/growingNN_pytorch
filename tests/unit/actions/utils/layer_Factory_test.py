import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.utils.layer_Factory import LinearFactory


"create_zero_linear sets weight and bias to all zeros"
def test_create_zero_linear_has_zero_weights_and_bias():
    layer = LinearFactory.create_zero_linear(in_features=5, out_features=7)
    assert torch.all(layer.weight == 0)
    assert torch.all(layer.bias == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
