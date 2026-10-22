"""Unit tests for CIFAR-10 loaders and the plain starter in train_cifar10."""

import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_PATH = _REPO_ROOT / "experiments" / "train_cifar10.py"


def _load_train_cifar10_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("train_cifar10", _EXPERIMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_cifar10(calls: dict[str, int]):
    def _factory(*_args, **kwargs):
        calls["count"] += 1
        n = 50_000 if kwargs.get("train", True) else 10_000
        return TensorDataset(torch.zeros(n, 1), torch.zeros(n, dtype=torch.long))

    return _factory


def test_prepare_loads_datasets_once(tmp_path, monkeypatch):
    """
    prepare should construct torchvision datasets only once per Cifar10Data instance.
    """
    # Arrange
    module = _load_train_cifar10_module()
    data = module.Cifar10Data(tmp_path)
    calls = {"count": 0}
    monkeypatch.setattr(module.datasets, "CIFAR10", _fake_cifar10(calls))
    (tmp_path / "cifar-10-batches-py").mkdir()
    (tmp_path / "cifar-10-batches-py" / "data_batch_1").write_bytes(b"x")

    # Act
    data.prepare()
    data.prepare()

    # Assert
    assert calls["count"] == 3
    assert data._datasets is not None


def test_loaders_reuses_cached_loaders_for_same_batch_size(tmp_path, monkeypatch):
    """
    loaders should return the same DataLoader objects for a repeated batch_size.
    """
    # Arrange
    module = _load_train_cifar10_module()
    data = module.Cifar10Data(tmp_path)
    monkeypatch.setattr(module.datasets, "CIFAR10", _fake_cifar10({"count": 0}))
    (tmp_path / "cifar-10-batches-py").mkdir()
    (tmp_path / "cifar-10-batches-py" / "data_batch_1").write_bytes(b"x")

    # Act
    train_loader_a, _, _ = data.loaders(4)
    train_loader_b, _, _ = data.loaders(4)

    # Assert
    assert train_loader_a is train_loader_b


def test_loaders_keep_official_test_out_of_the_training_tuple(tmp_path, monkeypatch):
    """
    loaders should return the 45k/5k train split, and test_loader should return the official 10k set.
    """
    # Arrange
    module = _load_train_cifar10_module()
    data = module.Cifar10Data(tmp_path)
    monkeypatch.setattr(module.datasets, "CIFAR10", _fake_cifar10({"count": 0}))
    (tmp_path / "cifar-10-batches-py").mkdir()
    (tmp_path / "cifar-10-batches-py" / "data_batch_1").write_bytes(b"x")

    # Act
    train_loader, val_loader, clean_loader = data.loaders(8)
    test_loader = data.test_loader(8)
    test_loader_again = data.test_loader(8)

    # Assert
    assert len(train_loader.dataset) == 45_000
    assert len(val_loader.dataset) == 5_000
    assert len(clean_loader.dataset) == 45_000
    assert len(test_loader.dataset) == 10_000
    assert test_loader is test_loader_again
    assert set(clean_loader.dataset.indices).isdisjoint(set(val_loader.dataset.indices))
    assert test_loader.dataset is not val_loader.dataset


def test_build_model_uses_grid_architecture_values():
    """
    _build_model should create the plain CIFAR starter from channel and hidden sizes in the grid entry.
    """
    # Arrange
    module = _load_train_cifar10_module()
    hyperparameters = {
        "model_channels": 8,
        "model_hidden_dim": 16,
    }

    # Act
    model = module._build_model(hyperparameters)

    # Assert
    assert model.conv1.in_channels == 3
    assert model.conv1.out_channels == 8
    assert model.linear.in_features == 8
    assert model.linear.out_features == 16
    assert model.linear2.out_features == 10
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.modules())


def test_plain_starter_outputs_ten_class_logits():
    """
    MinimalCifarNet should map a CIFAR-sized batch to one logit vector per image.
    """
    # Arrange
    module = _load_train_cifar10_module()
    model = module.MinimalCifarNet(channels=4, hidden_dim=8)
    inputs = torch.zeros(2, 3, 32, 32)

    # Act
    logits = model(inputs)

    # Assert
    assert logits.shape == (2, 10)
