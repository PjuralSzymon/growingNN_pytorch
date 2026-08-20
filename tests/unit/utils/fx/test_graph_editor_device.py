"""New modules inserted via ModelStructureEditor must match the GraphModule device."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_editor import _align_new_module_to_graph
from tests.model_factory import ModelFactory

_INPUT_SHAPE = (1, 4, 8, 8)


def test_align_new_module_to_graph_moves_cpu_layer_onto_cuda_graph():
    """
    _align_new_module_to_graph should place a CPU-built layer on the GraphModule device.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device-mismatch regression")

    # Arrange
    gm = TracedModel.create(ModelFactory.simple_conv_chain_2().cuda(), _INPUT_SHAPE).gm
    cpu_layer = nn.Conv2d(4, 4, 3, padding=1)

    # Act
    aligned = _align_new_module_to_graph(gm, cpu_layer)

    # Assert
    assert next(aligned.parameters()).device.type == "cuda"
    assert next(gm.parameters()).device.type == "cuda"


def test_add_res_conv_keeps_new_params_on_same_device_as_graph():
    """
    After AddResConvLayer.execute on a CUDA model, new residual weights must be CUDA
    so ShapeProp / generate_all_actions can deepen without device mismatch.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for device-mismatch regression")

    # Arrange
    device = torch.device("cuda")
    traced = TracedModel.create(ModelFactory.simple_conv_chain_2().to(device), _INPUT_SHAPE)
    actions = AddResConvLayer.generate_all_actions(traced)
    assert actions, "expected at least one residual conv action"
    action = actions[0]
    new_name = action.params[3]

    # Act
    action.execute(traced)
    new_param = next(traced.gm.get_submodule(new_name).parameters())
    deepen_actions = AddResConvLayer.generate_all_actions(traced)
    out = traced.gm(torch.randn(2, 4, 8, 8, device=device))

    # Assert
    assert new_param.device.type == "cuda"
    assert isinstance(deepen_actions, list)
    assert out.shape == (2, 4)


def test_add_res_conv_on_cpu_keeps_new_module_on_cpu():
    """
    CPU GraphModules should keep newly inserted residual modules on CPU.
    """
    # Arrange
    traced = TracedModel.create(ModelFactory.simple_conv_chain_2(), _INPUT_SHAPE)
    actions = AddResConvLayer.generate_all_actions(traced)
    assert actions, "expected at least one residual conv action"
    action = actions[0]
    new_name = action.params[3]

    # Act
    action.execute(traced)
    new_param = next(traced.gm.get_submodule(new_name).parameters())

    # Assert
    assert new_param.device.type == "cpu"
