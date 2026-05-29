"""Unit tests for ``growingnn.utils.fx_graph_drawer``."""

import sys
from pathlib import Path

import pytest
import torch.nn as nn
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils import fx_graph_drawer


def test_module_weight_shape_suffix_for_linear():
    """
    module_weight_shape_suffix should append weight shape for Linear modules.
    """
    # Arrange
    layer = nn.Linear(3, 5)

    # Act
    suffix = fx_graph_drawer.module_weight_shape_suffix(layer)

    # Assert
    assert suffix == "\\nweight (5, 3)"


def test_module_weight_shape_suffix_empty_for_relu():
    """
    module_weight_shape_suffix should return empty string for modules without .weight shape.
    """
    # Arrange / Act / Assert
    assert fx_graph_drawer.module_weight_shape_suffix(nn.ReLU()) == ""


def test_draw_filtered_fx_graph_renders_kept_nodes(monkeypatch):
    """
    draw_filtered_fx_graph should build a filtered dot graph and call render.
    """
    # Arrange
    gm = fx.symbolic_trace(nn.Linear(2, 2))
    rendered = []

    class FakeDot:
        def attr(self, *args, **kwargs):
            return None

        def node(self, *args, **kwargs):
            return None

        def edge(self, *args, **kwargs):
            return None

        def render(self, path, format, cleanup):
            rendered.append((path, format, cleanup))

    monkeypatch.setattr(fx_graph_drawer, "Digraph", lambda *args, **kwargs: FakeDot())

    # Act
    dot = fx_graph_drawer.draw_filtered_fx_graph(gm, output_file="tmp_filtered", fmt="svg")

    # Assert
    assert dot is not None
    assert rendered == [("tmp_filtered", "svg", True)]


def test_draw_torch_fx_graph_raises_on_unsupported_format():
    """
    draw_torch_fx_graph should raise ValueError when pydot has no writer for fmt.
    """
    # Arrange
    gm = fx.symbolic_trace(nn.Linear(2, 2))

    # Act / Assert
    with pytest.raises(ValueError, match="Unsupported graph format"):
        fx_graph_drawer.draw_torch_fx_graph(gm, fmt="not_a_real_format")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
