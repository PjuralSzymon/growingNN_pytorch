"""Unit tests for CIFAR-10 in-memory caching in train_cifar10."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXPERIMENT_PATH = _REPO_ROOT / "experiments" / "train_cifar10.py"


def _load_train_cifar10_module():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("train_cifar10", _EXPERIMENT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_loads_datasets_once(tmp_path, monkeypatch):
    """
    prepare should construct torchvision datasets only once per Cifar10Data instance.
    """
    # Arrange
    module = _load_train_cifar10_module()
    data = module.Cifar10Data(tmp_path)
    calls = {"count": 0}

    def _counting_cifar10(*args, **kwargs):
        calls["count"] += 1
        return MagicMock(__len__=lambda _self: 1)

    monkeypatch.setattr(module.datasets, "CIFAR10", _counting_cifar10)
    (tmp_path / "cifar-10-batches-py").mkdir()

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
    fake_dataset = MagicMock(__len__=lambda _self: 1)
    monkeypatch.setattr(
        module.datasets,
        "CIFAR10",
        lambda *_args, **_kwargs: fake_dataset,
    )
    (tmp_path / "cifar-10-batches-py").mkdir()

    # Act
    train_loader_a, _, _ = data.loaders(4)
    train_loader_b, _, _ = data.loaders(4)

    # Assert
    assert train_loader_a is train_loader_b


def test_build_model_uses_grid_architecture_values():
    """
    _build_model should create the CIFAR model from architecture values in the grid entry.
    """
    # Arrange
    module = _load_train_cifar10_module()
    hyperparameters = {
        "model_channels": 8,
        "model_hidden_dim": 16,
        "model_num_blocks": 2,
    }

    # Act
    model = module._build_model(hyperparameters)

    # Assert
    assert model.conv1.out_channels == 8
    assert model.linear.in_features == 16
    assert model.num_blocks == 2
