"""Unit tests for ``growingnn.actions.utils.fx_shape_probe``."""

import torch
import torch.fx as fx

from growingnn.actions.utils.fx_shape_probe import call_module_output_shapes
from tests.model_factory import ModelFactory


def test_call_module_output_shapes_records_conv_outputs():
    """
    After ``ShapeProp``, ``call_module_output_shapes`` should list a shape tuple
    for each traced ``Conv2d`` submodule (e.g. ``c1``, ``c2``).
    """

    # Arrange
    model = ModelFactory.simple_conv_chain_2()
    gm = fx.symbolic_trace(model)
    x = torch.randn(1, 4, 16, 16)

    # Act
    shapes = call_module_output_shapes(gm, x)

    # Assert
    assert "c1" in shapes and "c2" in shapes
    assert shapes["c1"] == (1, 4, 16, 16)
    assert shapes["c2"] == (1, 4, 16, 16)
