"""Unit tests for FX nary_add sum helpers."""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx.sum_nodes import connect_residual_branch, is_sum_node, nary_add


def test_nary_add_sums_multiple_tensors():
    """
    nary_add should return the elementwise sum of all input tensors.
    """
    # Arrange
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c = torch.tensor([5.0, 6.0])

    # Act
    result = nary_add(a, b, c)

    # Assert
    assert torch.equal(result, torch.tensor([9.0, 12.0]))


def test_connect_residual_branch_keeps_dst_in_sum_args():
    """
    connect_residual_branch should not replace dst inside the new nary_add args.
    """
    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)

        def forward(self, x):
            return self.l2(self.l1(x))

    gm = fx.symbolic_trace(Model())
    gm.add_module("res", nn.Linear(4, 4))
    nodes = list(gm.graph.nodes)
    l1 = next(node for node in nodes if node.op == "call_module" and node.target == "l1")
    l2 = next(node for node in nodes if node.op == "call_module" and node.target == "l2")

    # Act
    connect_residual_branch(gm, l2, l1, "res")
    gm.graph.lint()
    gm.recompile()
    nary_node = next(node for node in gm.graph.nodes if node.op == "call_function" and node.target is nary_add)

    # Assert
    assert l2 in nary_node.args
    assert nary_node not in nary_node.args


def test_connect_residual_branch_flattens_existing_binary_add():
    """
    connect_residual_branch should replace nested adds with one nary_add node.
    """
    # Arrange
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(4, 4)
            self.l2 = nn.Linear(4, 4)
            self.l3 = nn.Linear(4, 4)

        def forward(self, x):
            a = self.l1(x)
            b = self.l2(a)
            return a + self.l3(b)

    gm = fx.symbolic_trace(Model())
    gm.add_module("branch", nn.Linear(4, 4))
    gm.branch.weight.data.zero_()
    gm.branch.bias.data.zero_()
    existing_sum = next(node for node in gm.graph.nodes if is_sum_node(node))
    dst = existing_sum.all_input_nodes[1]
    src = existing_sum.all_input_nodes[0]

    # Act
    connect_residual_branch(gm, dst, src, "branch")
    gm.graph.lint()
    gm.recompile()
    nary_nodes = [node for node in gm.graph.nodes if node.op == "call_function" and node.target is nary_add]

    # Assert
    assert len(nary_nodes) == 1
    assert len(nary_nodes[0].args) == 3
