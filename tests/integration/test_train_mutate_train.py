"""Integration: train ResNet-18, mutate architecture, train again."""

import sys
from pathlib import Path

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.core.config
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph
from growingnn.actions.add_seq_conv_layer import AddSeqConvLayer
from growingnn.actions.add_seq_layer import AddSeqLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.training.gradient_descent import gradient_descent
from growingnn.training.lr_scheduler import LearningRateScheduler, ScheduleMode
from growingnn.training.stoppers import StopperMode, TrainingStopper
from growingnn.utils.fx import GraphStructureQuery
from tests.regression.regression_utils import FOLDER_NAME


def _loaders(seed: int = 0, n: int = 64, batch_size: int = 8, num_classes: int = 2):
    torch.manual_seed(seed)
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, num_classes, (n,))
    train = DataLoader(TensorDataset(x[:48], y[:48]), batch_size=batch_size, shuffle=True)
    val = DataLoader(TensorDataset(x[48:], y[48:]), batch_size=batch_size)
    return train, val


def _trace_resnet(num_classes: int = 2) -> fx.GraphModule:
    model = resnet18(weights=None, num_classes=num_classes)
    return fx.symbolic_trace(model)


def _train(
    gm: fx.GraphModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 1,
) -> dict[str, list[float]]:
    _, history = gradient_descent(
        gm,
        epochs,
        train_loader,
        val_loader,
        nn.CrossEntropyLoss(),
        LearningRateScheduler(ScheduleMode.CONSTANT, alpha=0.01),
        TrainingStopper(StopperMode.EMPTY),
        quiet=True,
    )
    return history


def _first_grow_action(gm: fx.GraphModule):
    actions = AddSeqConvLayer.generate_all_actions(gm)
    if not actions:
        actions = AddSeqLayer.generate_all_actions(gm)
    assert actions, "expected at least one grow action on ResNet-18"
    return actions[0]


def test_train_grow_train_resnet18_still_learns():
    """
    ResNet-18 should keep learning after AddSeqConvLayer grows the traced graph.
    """
    # Arrange
    torch.manual_seed(0)
    gm = _trace_resnet()
    train_loader, val_loader = _loaders()
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)

    # Act
    history_before = _train(gm, train_loader, val_loader, epochs=1)
    _first_grow_action(gm).execute(gm)
    params_after_grow = GraphStructureQuery.get_amount_of_parameters(gm)
    history_after = _train(gm, train_loader, val_loader, epochs=1)

    # Assert
    assert params_after_grow > params_before
    assert history_before["train_loss"][-1] < history_before["train_loss"][0]
    assert history_after["train_loss"][-1] < history_after["train_loss"][0]
    assert history_after["val_acc"][-1] >= history_after["val_acc"][0]


def test_train_shrink_train_resnet18_still_learns():
    """
    ResNet-18 should keep learning after DelLayer shrinks the traced graph.
    """
    # Arrange
    torch.manual_seed(0)
    gm = _trace_resnet()
    train_loader, val_loader = _loaders()
    params_before = GraphStructureQuery.get_amount_of_parameters(gm)

    # Act
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified_init", fmt="pdf")
    history_before = _train(gm, train_loader, val_loader, epochs=1)
    shrink_actions = DelLayer.generate_all_actions(gm)
    assert shrink_actions, "expected at least one shrink action on ResNet-18"
    shrink_actions[0].execute(gm)
    print(f"shrink action executed: {shrink_actions[0]}")
    draw_filtered_fx_graph(gm, FOLDER_NAME + "/" + "fx_graph_simplified_shrink", fmt="pdf")
    params_after_shrink = GraphStructureQuery.get_amount_of_parameters(gm)
    history_after = _train(gm, train_loader, val_loader, epochs=1)

    # Assert
    assert params_after_shrink < params_before
    assert history_before["train_loss"][-1] < history_before["train_loss"][0]
    assert history_after["train_loss"][-1] < history_after["train_loss"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
