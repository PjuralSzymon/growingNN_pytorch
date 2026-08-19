"""Unit tests for ``growingnn.utils.fx.graph_extraction``."""

import sys
from pathlib import Path

import pytest
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx.graph_extraction import extract_graph
from tests.model_factory import ModelFactory


def test_extract_graph_returns_existing_graphmodule():
    """
    extract_graph should return the same GraphModule instance when already traced.
    """

    # Arrange
    gm = fx.symbolic_trace(ModelFactory.simple_chain_2())

    # Act
    result = extract_graph(gm)

    # Assert
    assert result is gm


def test_extract_graph_traces_linear_chain():
    """
    extract_graph should produce a GraphModule with Linear call_module nodes.
    """

    # Arrange
    model = ModelFactory.simple_chain_2()

    # Act
    gm = extract_graph(model)

    # Assert
    targets = [n.target for n in gm.graph.nodes if n.op == "call_module"]
    assert targets == ["l1", "l2"]


def test_extract_graph_raises_original_error_when_huggingface_unavailable(monkeypatch):
    """
    extract_graph should re-raise the torch.fx error when HuggingFace FX is not used.
    """

    # Arrange
    class Broken(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x
            return x + 1

    import growingnn.utils.fx.graph_extraction as ge

    monkeypatch.setattr(ge, "_huggingface_trace", lambda *args, **kwargs: None)
    model = Broken()

    # Act / Assert
    with pytest.raises(fx.proxy.TraceError, match="control flow"):
        extract_graph(model)


def test_extract_graph_uses_huggingface_when_input_names_given(monkeypatch):
    """
    extract_graph should call HuggingFace FX when input_names is set, not torch.fx.
    """

    # Arrange
    model = ModelFactory.simple_chain_2()
    seen: list[list[str] | None] = []

    def fake_hf(m, attempts):
        seen.extend(attempts)
        return fx.symbolic_trace(m)

    import growingnn.utils.fx.graph_extraction as ge

    monkeypatch.setattr(ge, "_huggingface_trace", fake_hf)

    # Act
    gm = extract_graph(model, input_names=["inputs_embeds"])

    # Assert
    assert seen == [["inputs_embeds"]]
    assert isinstance(gm, fx.GraphModule)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
