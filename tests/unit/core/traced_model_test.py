"""Unit tests for TracedModel."""

import torch
import torch.fx as fx

from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser
from tests.model_factory import ModelFactory


def test_shapes_lazy_cache_runs_shapeprop_once():
    """
    shapes() should compute layer maps once and reuse them until invalidate().
    """
    # Arrange
    traced = TracedModel.create(fx.symbolic_trace(ModelFactory.simple_chain_2()), (1, 4))
    calls = {"count": 0}
    original = LayerShapeAnalyser.collect_layer_shapes

    def counting_collect(gm, example=None, *, input_shape=None):
        calls["count"] += 1
        return original(gm, example, input_shape=input_shape)

    LayerShapeAnalyser.collect_layer_shapes = staticmethod(counting_collect)

    # Act
    first = traced.shapes()
    second = traced.shapes()
    LayerShapeAnalyser.collect_layer_shapes = original

    # Assert
    assert calls["count"] == 1
    assert first == second
    assert "l1" in first[0]


def test_invalidate_forces_shape_recompute():
    """
    invalidate() should clear cached shapes so the next shapes() recomputes.
    """
    # Arrange
    traced = TracedModel.create(fx.symbolic_trace(ModelFactory.simple_chain_2()), (1, 4))
    traced.shapes()

    # Act
    traced.invalidate()
    outputs, _ = traced.shapes()

    # Assert
    assert "l1" in outputs


def test_create_uses_batch_one_input_shape():
    """
    create should store batch-1 input shape derived from a training batch.
    """
    # Arrange
    batch = torch.randn(8, 4)

    # Act
    traced = TracedModel.create(
        ModelFactory.simple_chain_2(),
        tuple(int(x) for x in batch[0:1].shape),
    )

    # Assert
    assert traced.input_shape == (1, 4)


def test_action_execute_invalidates_traced_model_cache():
    """
    Action.execute should clear TracedModel caches after mutating the graph.
    """
    # Arrange
    from growingnn.actions.add_neurons import AddNeurons

    traced = TracedModel.create(fx.symbolic_trace(ModelFactory.simple_chain_2()), (1, 4))
    traced.shapes()

    # Act
    AddNeurons(["l1", 1.5]).execute(traced)

    # Assert
    assert traced._out_shapes is None
    assert traced._in_shapes is None
