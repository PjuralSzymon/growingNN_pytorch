"""Trace an nn.Module: torch.fx first, HuggingFace FX if that fails."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn

from growingnn.core.logger import logger


def extract_graph(
    model: nn.Module | fx.GraphModule,
    input_names: list[str] | None = None,
) -> fx.GraphModule:
    """Return a GraphModule. Already traced graphs are returned as-is.

    Without input_names: torch.fx.symbolic_trace, then HuggingFace FX.
    With input_names: HuggingFace FX only (torch.fx cannot use those names).
    """
    if isinstance(model, fx.GraphModule):
        return model
    if input_names is not None:
        gm = _huggingface_trace(model, [input_names])
        if gm is not None:
            logger.info("extract_graph used HuggingFace FX with Conv1D leaves")
            return gm
        return fx.symbolic_trace(model)
    try:
        return fx.symbolic_trace(model)
    except Exception as fx_err:
        gm = _huggingface_trace(
            model,
            [["inputs_embeds"], ["pixel_values"], None],
        )
        if gm is not None:
            logger.info("extract_graph used HuggingFace FX with Conv1D leaves")
            return gm
        raise fx_err


def _huggingface_trace(
    model: nn.Module,
    attempts: list[list[str] | None],
) -> fx.GraphModule | None:
    """HF symbolic_trace with Conv1D leaves, or None if transformers.utils.fx is missing."""
    try:
        from transformers.utils.fx import HFTracer, symbolic_trace as hf_symbolic_trace
    except ModuleNotFoundError:
        return None

    class Conv1DLeafHFTracer(HFTracer):
        def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
            if type(m).__name__ == "Conv1D":
                return True
            return super().is_leaf_module(m, module_qualified_name)

    last_err: Exception | None = None
    for names in attempts:
        try:
            kwargs: dict = {"tracer_cls": Conv1DLeafHFTracer}
            if names is not None:
                kwargs["input_names"] = names
            return hf_symbolic_trace(model, **kwargs)
        except Exception as err:
            last_err = err
            continue
    if last_err is not None:
        logger.debug("HuggingFace FX failed: %s: %s", type(last_err).__name__, last_err)
    return None
