"""
Integration: DelNeurons.generate_all_actions then execute on a residual model with norms.

Uses live ``PASSTHROUGH_MODULES`` from config (no monkeypatch). Fails if BatchNorm is listed
as passthrough while neuron shrink must resize BatchNorm along the graph path.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.delete_neurons import DelNeurons
from growingnn.utils.fx import ModuleResolver, NodeWidthAnalyser
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph, draw_torch_fx_graph
from tests.regression.regression_utils import FOLDER_NAME, clear_regression_folder, parse_regression_cli
from growingnn.core.traced_model import TracedModel
BATCH_SIZE = 2
INPUT_FEATURES = 4
HIDDEN_WIDTH = 100
OUTPUT_FEATURES = 4
FILE_PATH = "testResults/integration/del_neurons_passthrough"


class _PassthroughForkResidual(nn.Module):
    """
    in -> hidden1 -> [passthrough -> hidden] x N branches -> add -> output.

    Each branch uses its own Linear hidden layer then a different PASSTHROUGH_MODULES type.
    All branch outputs are summed, then mapped to the output layer.
    """

    def __init__(self, in_features: int = INPUT_FEATURES, width: int = HIDDEN_WIDTH, out_features: int = OUTPUT_FEATURES):
        super().__init__()
        self.hidden1 = nn.Linear(in_features, width)

        self.relu = nn.ReLU()
        self.hidden_relu = nn.Linear(width, width)

        self.bn = nn.BatchNorm1d(width)
        self.hidden_bn = nn.Linear(width, width)

        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.hidden_gelu = nn.Linear(width, width)

        self.leaky = nn.LeakyReLU(0.1)
        self.hidden_leaky = nn.Linear(width, width)

        self.silu = nn.SiLU()
        self.hidden_silu = nn.Linear(width, width)

        self.tanh = nn.Tanh()
        self.hidden_tanh = nn.Linear(width, width)

        self.elu = nn.ELU()
        self.hidden_elu = nn.Linear(width, width)

        self.sigmoid = nn.Sigmoid()
        self.hidden_sigmoid = nn.Linear(width, width)

        self.identity = nn.Identity()
        self.hidden_id = nn.Linear(width, width)

        self.avgpool = nn.AvgPool1d(kernel_size=1)
        self.hidden_avgpool = nn.Linear(width, width)

        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.hidden_maxpool = nn.Linear(width, width)

        self.adaptive_avgpool = nn.AdaptiveAvgPool1d(1)
        self.hidden_adaptive_avgpool = nn.Linear(width, width)

        self.adaptive_maxpool = nn.AdaptiveMaxPool1d(1)
        self.hidden_adaptive_maxpool = nn.Linear(width, width)

        self.output = nn.Linear(width, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden1(x)

        b_relu = self.relu(self.hidden_relu(h))
        b_bn = self.bn(self.hidden_bn(h))
        b_gelu = self.gelu(self.drop(self.hidden_gelu(h)))
        b_leaky = self.leaky(self.hidden_leaky(h))
        b_silu = self.silu(self.hidden_silu(h))
        b_tanh = self.tanh(self.hidden_tanh(h))
        b_elu = self.elu(self.hidden_elu(h))
        b_sigmoid = self.sigmoid(self.hidden_sigmoid(h))
        b_id = self.identity(self.hidden_id(h))
        b_avgpool = self.avgpool(self.hidden_avgpool(h).unsqueeze(-1)).squeeze(-1)
        b_maxpool = self.maxpool(self.hidden_maxpool(h).unsqueeze(-1)).squeeze(-1)
        b_adaptive_avg = self.adaptive_avgpool(self.hidden_adaptive_avgpool(h).unsqueeze(-1)).squeeze(-1)
        b_adaptive_max = self.adaptive_maxpool(self.hidden_adaptive_maxpool(h).unsqueeze(-1)).squeeze(-1)

        merged = (
            b_relu + b_bn + b_gelu + b_leaky + b_silu + b_tanh + b_elu + b_sigmoid + b_id
            + b_avgpool + b_maxpool + b_adaptive_avg + b_adaptive_max
        )
        return self.output(merged)

def _passthrough_fork_residual_model() -> nn.Module:
    return _PassthroughForkResidual()

def test_del_neurons_generate_then_execute_aligns_widths_on_residual_model(save_output: bool = False):
    """
    For each DelNeurons action from generate_all_actions on the passthrough-fork residual
    model, execute must keep module widths consistent and forward must succeed.
    """
    # Arrange
    probe = _passthrough_fork_residual_model()
    gm = fx.symbolic_trace(probe)
    x = torch.randn(BATCH_SIZE, INPUT_FEATURES)
    actions = DelNeurons.generate_all_actions(TracedModel.create(gm, (1, 4)))
    output_1 = gm(x)
    if output_1 is None:
        raise ValueError("output is None")
    
    if save_output:
        draw_filtered_fx_graph(gm, FILE_PATH, fmt="pdf")

    # Act / Assert
    i = 0
    for action in actions:
        action.execute(TracedModel.create(gm, (1, 4)))
        print(action)
        if save_output:
            draw_filtered_fx_graph(gm, FILE_PATH + str(i), fmt="pdf")
            draw_torch_fx_graph(gm, FILE_PATH + "_complex" + str(i), fmt="pdf")
        out = gm(x)
        action_used = action.params[0]
        output_layer = gm.output
        print(
            f"action_used: {action_used}, out_shape: {tuple(out.shape)}, "
            f"output_layer: ({output_layer.in_features}, {output_layer.out_features})"
        )
        i = i + 1

if __name__ == "__main__":
    args = parse_regression_cli()
    test_del_neurons_generate_then_execute_aligns_widths_on_residual_model(save_output=args.save_output)
    if not args.save_output:
        clear_regression_folder()
