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
from tests.model_factory import ModelFactory
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.core.traced_model import TracedModel



def test_add_seq_linear_layer_generate_all_actions_linear_chain():
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)

    actions = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))

    # For l1->l2->l3, module_dependency_pairs yields 3 pairs; we create one action per Layer_Type.
    assert len(actions) == len(list(Layer_Type)) - 1


def test_large_linear_chain_uses_bounded_fx_edge_candidates():
    model = torch.nn.Sequential(*(torch.nn.Linear(4, 4) for _ in range(300))).eval()
    traced = TracedModel.create(model, (1, 4))

    actions = AddSeqLinearLayer.generate_all_actions(traced)

    # Exhaustive source-module reachability would inspect every module/graph
    # pair. Large graphs use the 299 adjacent activation seams instead.
    assert len(actions) == 299
    assert all(action.params[4] == "fx_edge" for action in actions)

def test_add_seq_linear_layer_execute():
    #Arrange
    model = ModelFactory.simple_chain_3()
    gm = fx.symbolic_trace(model)
    executed_actions = []
    x = torch.randn(2, 4)
    rng = random.Random(42)
    initial_amount_of_linears = sum(1 for m in gm.modules() if isinstance(m, torch.nn.Linear))

    # Act
    for _ in range(30):
        actions: List[AddSeqLinearLayer] = AddSeqLinearLayer.generate_all_actions(TracedModel.create(gm, (1, 4)))
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


def test_add_seq_linear_layer_generates_and_executes_fx_edge_action_on_rank_three():
    class FunctionalSequence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(self, x):
            return torch.relu(self.proj(x) + x)

    gm = fx.symbolic_trace(FunctionalSequence()).eval()
    x = torch.randn(2, 3, 4)
    expected = gm(x)
    traced = TracedModel.create(gm, (2, 3, 4))
    actions = AddSeqLinearLayer.generate_all_actions(traced)
    edge_action = next(action for action in actions if len(action.params) > 4 and action.params[4] == "fx_edge")

    edge_action.execute(traced)

    torch.testing.assert_close(gm(x), expected)
    assert any(node.op == "call_module" and node.target == edge_action.params[3] for node in gm.graph.nodes)


def test_add_seq_linear_layer_reports_rank_four_as_unknown_feature_axis():
    class ImageActivation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, kernel_size=1)

        def forward(self, x):
            return torch.relu(self.conv(x))

    gm = fx.symbolic_trace(ImageActivation())
    actions = AddSeqLinearLayer.generate_all_actions(
        TracedModel.create(gm, (1, 2, 8, 8))
    )

    assert not any(len(action.params) > 4 and action.params[4] == "fx_edge" for action in actions)


def test_add_seq_linear_layer_accepts_explicit_channels_first_feature_axis():
    class ImageSequence(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=1)
            self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=1)

        def forward(self, x):
            return self.conv2(torch.relu(self.conv1(x)))

    gm = fx.symbolic_trace(ImageSequence()).eval()
    x = torch.randn(2, 3, 8, 8)
    expected = gm(x)
    traced = TracedModel.create(gm, tuple(x.shape), example_args=(x,))
    actions = AddSeqLinearLayer.generate_all_actions(
        traced, feature_axis=1, supported_ranks=(4,)
    )
    action = next(
        item for item in actions
        if len(item.params) > 4 and item.params[4] == "fx_edge"
    )
    assert action.params[2].feature_axis == 1

    action.execute(traced)

    torch.testing.assert_close(gm(x), expected)




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
