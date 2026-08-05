"""AddSeqConvLayer before-flatten path on MNIST medium / boundary stems."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.core.traced_model import TracedModel
from experiments.train_mnist_exp001_slope_model_depth import MediumMnistNet


def _conv2d_modules(gm: fx.GraphModule) -> list[str]:
    return [name for name, mod in gm.named_modules() if isinstance(mod, nn.Conv2d) and name]


def test_medium_mnist_generates_seq_conv_before_flatten():
    """
    MediumMnistNet should emit an AddSeqConvLayer on conv1→linear.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))

    # Act
    actions = AddSeqConvLayer.generate_all_actions(traced)

    # Assert
    assert len(actions) >= 1
    action = actions[0]
    assert action.params[0] == "conv1"
    assert action.params[1] == "linear"
    assert isinstance(action.params[2], nn.Conv2d)
    assert len(action.params) == 4


def test_medium_mnist_execute_inserts_second_conv_before_flatten():
    """
    Executing before-flatten seq conv on Medium should add a Conv2d feeding flatten.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))
    actions = AddSeqConvLayer.generate_all_actions(traced)
    action = actions[0]
    linear_in_before = traced.gm.linear.in_features

    # Act
    action.execute(traced)
    out = traced.gm(torch.randn(2, 1, 28, 28))
    new_name = action.params[3]
    new_node = next(n for n in traced.gm.graph.nodes if n.op == "call_module" and n.target == new_name)

    # Assert
    assert len(_conv2d_modules(traced.gm)) == 2
    assert any(u.op == "call_method" and u.target == "flatten" for u in new_node.users)
    assert traced.gm.get_submodule("linear").in_features == linear_in_before
    assert out.shape == (2, 10)


def test_linear_to_linear_pair_never_yields_seq_conv():
    """
    AddSeqConvLayer must not propose actions whose endpoints are both Linear.
    """
    # Arrange
    traced = TracedModel.create(MediumMnistNet(), (1, 1, 28, 28))

    # Act
    actions = AddSeqConvLayer.generate_all_actions(traced)

    # Assert
    for action in actions:
        from_mod = traced.gm.get_submodule(action.params[0])
        to_mod = traced.gm.get_submodule(action.params[1])
        assert not (isinstance(from_mod, nn.Linear) and isinstance(to_mod, nn.Linear))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
