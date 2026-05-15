import sys
from pathlib import Path

# Repo root must be on sys.path before any `growingnn` / `tests` imports (script may be run from any cwd).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import random
from typing import List

import torch
import torch.fx as fx
from torch.fx.passes.graph_drawer import FxGraphDrawer

from growingnn.actions.action import Layer_Type
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_res_layer import AddResLayer
from growingnn.actions.utils.model_analyser import get_amount_of_parameters
from growingnn.core.logger import logger
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.model_factory import ModelFactory
from tests.regression.regression_utils import (
    FOLDER_NAME,
    clear_regression_folder,
    parse_regression_cli,
    plot_norms_and_parameter_count,
)

import torch.nn as nn


class InnerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(4, 4)
        self.act = nn.ReLU()
        self.l2 = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x

class MiddleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = InnerBlock()
        self.l1 = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.inner(x)
        x = self.l1(x)
        x = self.act(x)
        return x

class OuterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.middle = MiddleBlock()
        self.l1 = nn.Linear(4, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.middle(x)
        x = self.l1(x)
        x = self.act(x)
        return x

class ModelDeeplyNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Linear(4, 4)
        self.outer = OuterBlock()
        self.head = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.stem(x)
        x = self.outer(x)
        x = self.head(x)
        return x



if __name__ == "__main__":
    args = parse_regression_cli()
    model = ModelDeeplyNested()
    model.eval()
    gm = fx.symbolic_trace(model)

    # Act 
    FOLDER_NAME = "testResults/regression"
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "graph_unit_test_filtered", fmt="pdf")
    draw_torch_fx_graph(gm, FOLDER_NAME + "/" + "graph_unit_test", fmt="pdf")
