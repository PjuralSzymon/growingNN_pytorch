import random
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_linear_layer import AddResLinearLayer
from growingnn.core import config
from tests.model_factory import ModelFactory
from growingnn.core.traced_model import TracedModel


def test_generate_all_actions_skips_when_weight_matrix_exceeds_config_limit(monkeypatch):
    """
    AddResLinearLayer should skip pairs whose Linear weight matrix in*out exceeds config limit.
    """

    # Arrange
    monkeypatch.setattr(config, "MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE", 1)
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    # Act
    actions = AddResLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # Assert
    assert actions == []


"Generate AddResLinearLayer actions for a simple linear chain"
def test_add_res_linear_layer_generate_all_actions_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    actions = AddResLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # For l1->l2->l3, only l1->l2 is an edge into a hidden module; one action per Layer_Type.
    assert len(actions) == len(list(Layer_Type))


def test_residual_pair_compatibility_path_rejects_distant_module_endpoints():
    gm = fx.symbolic_trace(ModelFactory.simple_chain_3())

    assert AddResLinearLayer._module_pair_is_local(gm, "l1", "l2")
    assert not AddResLinearLayer._module_pair_is_local(gm, "l1", "l3")

def test_add_res_linear_layer_execute():
    #Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    initial_amount_of_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))

    # Act
    for _ in range(30):
        actions: List[AddResLinearLayer] = AddResLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))
        idx = rng.randrange(len(actions))
        actions[idx].execute(TracedModel.create(gm, (1, 4)))
    out = gm(x)

    # Assert
    num_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))
    assert num_linears == 30 + initial_amount_of_linears
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
    for action in executed_actions:
        assert action.params[0] in gm.graph.nodes
        assert action.params[1] in gm.graph.nodes


def test_add_res_linear_layer_generates_fx_merge_action_for_rank_three():
    class FunctionalResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x) + x

    gm = fx.symbolic_trace(FunctionalResidual()).eval()
    x = torch.randn(2, 3, 4)
    expected = gm(x)
    traced = TracedModel.create(gm, (2, 3, 4))
    actions = AddResLinearLayer.generate_all_actions(
        traced, layer_types=(Layer_Type.ZERO,)
    )
    action = next(
        item for item in actions
        if len(item.params) > 4 and item.params[4] == "fx_residual"
    )

    action.execute(traced)

    torch.testing.assert_close(gm(x), expected)
    assert any(node.op == "call_module" and node.target == action.params[3] for node in gm.graph.nodes)


def test_add_res_linear_layer_recognizes_exported_aten_add():
    class ExportResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.proj(x) + x

    x = torch.randn(2, 4)
    gm = TracedModel.create(
        ExportResidual().eval(), (2, 4), capture="export", example_args=(x,)
    ).gm
    expected = gm(x)
    traced = TracedModel.create(gm, (2, 4))
    actions = AddResLinearLayer.generate_all_actions(
        traced, layer_types=(Layer_Type.ZERO,)
    )
    action = next(
        item for item in actions
        if len(item.params) > 4 and item.params[4] == "fx_residual"
    )

    action.execute(traced)

    torch.testing.assert_close(gm(x), expected)


def test_add_res_linear_layer_accepts_explicit_channels_first_feature_axis():
    class ImageResidual(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, kernel_size=1)

        def forward(self, x):
            return self.conv(x) + x

    gm = fx.symbolic_trace(ImageResidual()).eval()
    x = torch.randn(2, 4, 8, 8)
    expected = gm(x)
    traced = TracedModel.create(gm, tuple(x.shape), example_args=(x,))
    actions = AddResLinearLayer.generate_all_actions(
        traced,
        layer_types=(Layer_Type.ZERO,),
        feature_axis=1,
        supported_ranks=(4,),
    )
    action = next(
        item for item in actions
        if len(item.params) > 4 and item.params[4] == "fx_residual"
    )
    assert action.params[2].feature_axis == 1

    action.execute(traced)

    torch.testing.assert_close(gm(x), expected)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
