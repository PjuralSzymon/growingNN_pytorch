import torch
import torch.fx as fx

from growingnn.actions.residual_block import ResidualBlockAction
from growingnn.core.traced_model import TracedModel
from tests.model_factory import ModelFactory


def _traced():
    return TracedModel.create(fx.symbolic_trace(ModelFactory.residual_skip()), (2, 4))


def test_residual_skip_discovers_duplicate_and_keeps_the_last_block():
    actions = ResidualBlockAction.generate_all_actions(_traced())
    assert [action.params[0] for action in actions] == ["duplicate"]


def test_duplicate_residual_region_copies_parameters_and_keeps_output_shape():
    traced = _traced()
    probe = torch.randn(2, 4)
    expected_shape = traced.gm(probe).shape
    before = sum(parameter.numel() for parameter in traced.gm.parameters())

    ResidualBlockAction.generate_all_actions(traced, shrink=False)[0].execute(traced)

    traced.gm.graph.lint()
    assert sum(parameter.numel() for parameter in traced.gm.parameters()) > before
    assert traced.gm(probe).shape == expected_shape


def test_shrink_after_duplicate_keeps_output_shape_and_graph_valid():
    traced = _traced()
    probe = torch.randn(2, 4)
    expected_shape = traced.gm(probe).shape
    ResidualBlockAction.generate_all_actions(traced, shrink=False)[0].execute(traced)
    traced.invalidate()

    ResidualBlockAction.generate_all_actions(traced, grow=False)[0].execute(traced)

    traced.gm.graph.lint()
    assert traced.gm(probe).shape == expected_shape
