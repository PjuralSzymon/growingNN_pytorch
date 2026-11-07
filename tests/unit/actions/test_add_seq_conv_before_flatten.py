"""AddSeqConvLayer before-flatten path on MNIST medium / boundary stems."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F

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
    assert len(action.params) == 5
    assert action.params[4] is True


def test_medium_mnist_execute_inserts_second_conv_before_flatten():
    """
    Executing before-flatten seq conv on Medium should add a Conv2d ahead of the global pool.
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
    assert any(
        u.op == "call_function" and getattr(u.target, "__name__", "") == "adaptive_avg_pool2d"
        for u in new_node.users
    )
    assert traced.gm.get_submodule("linear").in_features == linear_in_before
    assert out.shape == (2, 10)


class _GlobalPoolStem(nn.Module):
    """Conv, windowed pool, global pool, then a two-layer head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.linear = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.adaptive_avg_pool2d(x, 1)
        x = F.relu(self.linear(x.flatten(1)))
        return self.linear2(x)


def _call_names(gm: fx.GraphModule) -> list[str]:
    names: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            names.append(str(node.target))
        elif node.op == "call_function":
            names.append(getattr(node.target, "__name__", str(node.target)))
    return names


def test_seq_conv_before_flatten_inserts_ahead_of_global_pool():
    """
    A before-flatten sequential conv should run before a trailing global pool.
    """
    # Arrange
    traced = TracedModel.create(_GlobalPoolStem(), (1, 3, 32, 32))
    action = AddSeqConvLayer.generate_all_actions(traced)[0]

    # Act
    action.execute(traced)

    # Assert
    names = _call_names(traced.gm)
    assert names.index(action.params[3]) < names.index("adaptive_avg_pool2d")


def test_seq_conv_before_flatten_can_stack_a_second_conv():
    """
    After one before-flatten conv, search should still offer another conv ahead of the global pool.
    """
    # Arrange
    traced = TracedModel.create(_GlobalPoolStem(), (1, 3, 32, 32))
    first = AddSeqConvLayer.generate_all_actions(traced)[0]
    first.execute(traced)

    # Act
    second = AddSeqConvLayer.generate_all_actions(traced)[0]
    second.execute(traced)

    # Assert
    names = _call_names(traced.gm)
    pool_at = names.index("adaptive_avg_pool2d")
    assert names.index(first.params[3]) < pool_at
    assert names.index(second.params[3]) < pool_at
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_seq_conv_bn_relu_preserves_eval_output_and_stacks():
    """
    A BatchNorm-ReLU sequential conv should keep eval logits and still allow another conv.
    """
    # Arrange
    traced = TracedModel.create(_GlobalPoolStem(), (1, 3, 32, 32))
    sample = torch.randn(2, 3, 32, 32)
    traced.gm.eval()
    before = traced.gm(sample).detach()
    first = AddSeqConvLayer.generate_all_actions(traced, add_batchnorm_and_relu=True)[0]

    # Act
    first.execute(traced)
    traced.gm.eval()
    after = traced.gm(sample).detach()

    # Assert
    bn_name = f"{first.params[3]}_bn"
    bn_node = next(n for n in traced.gm.graph.nodes if n.op == "call_module" and n.target == bn_name)
    relu_node = next(
        user for user in bn_node.users
        if user.op == "call_function" and getattr(user.target, "__name__", "") == "relu"
    )
    assert any(
        getattr(user.target, "__name__", "") == "adaptive_avg_pool2d" for user in relu_node.users
    )
    assert torch.allclose(before, after, atol=1e-4)
    second = AddSeqConvLayer.generate_all_actions(traced, add_batchnorm_and_relu=True)[0]
    second.execute(traced)
    assert second.params[5] is True
    assert traced.gm(sample).shape == (2, 10)


def test_seq_conv_can_stay_after_the_global_pool():
    """
    before_global_pool=False should insert the sequential conv after the trailing global pool.
    """
    # Arrange
    traced = TracedModel.create(_GlobalPoolStem(), (1, 3, 32, 32))
    action = AddSeqConvLayer.generate_all_actions(traced, before_global_pool=False)[0]

    # Act
    action.execute(traced)

    # Assert
    names = _call_names(traced.gm)
    assert names.index("adaptive_avg_pool2d") < names.index(action.params[3])
    assert action.params[6] is False


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
