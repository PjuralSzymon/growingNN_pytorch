"""Test-only torch.fx tracer that keeps a module named Conv1D as a leaf."""

from __future__ import annotations

import torch.fx as fx
import torch.nn as nn


class Conv1DLeafTracer(fx.Tracer):
    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        if type(m).__name__ == "Conv1D":
            return True
        return super().is_leaf_module(m, module_qualified_name)


def trace_conv1d_leaves(model: nn.Module) -> fx.GraphModule:
    return fx.GraphModule(model, Conv1DLeafTracer().trace(model))
