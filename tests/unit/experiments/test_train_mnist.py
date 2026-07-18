"""Unit tests for the minimal MNIST experiment."""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from experiments import train_mnist


class _FakeMNIST(Dataset):
    """Small deterministic replacement for torchvision MNIST."""

    calls: list[dict[str, object]] = []

    def __init__(self, root, train, download, transform):
        self.train = train
        self.calls.append({
            "root": root,
            "train": train,
            "download": download,
            "transform": transform,
        })

    def __len__(self) -> int:
        return 60_000 if self.train else 10_000

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return torch.zeros(1, 28, 28), index


def test_mnist_data_prepare_builds_augmented_clean_and_validation_sets(
    tmp_path, monkeypatch
):
    """MNISTData should build augmented training, clean training, and validation sets."""
    # Arrange
    _FakeMNIST.calls.clear()
    monkeypatch.setattr(train_mnist.datasets, "MNIST", _FakeMNIST)
    data = train_mnist.MNISTData(tmp_path, num_workers=0)

    # Act
    data.prepare()

    # Assert
    train, val, clean_train = data._datasets
    assert [call["train"] for call in _FakeMNIST.calls] == [True, True]
    assert len(train) == train_mnist.TRAIN_SIZE
    assert len(val) == train_mnist.VAL_SIZE
    assert train.indices == clean_train.indices
    assert set(train.indices).isdisjoint(val.indices)
    assert isinstance(_FakeMNIST.calls[0]["transform"].transforms[0], transforms.RandomAffine)


def test_mnist_data_loads_test_only_when_requested(tmp_path, monkeypatch):
    """MNISTData should keep the official test split out of training loaders."""
    # Arrange
    _FakeMNIST.calls.clear()
    monkeypatch.setattr(train_mnist.datasets, "MNIST", _FakeMNIST)
    data = train_mnist.MNISTData(tmp_path, num_workers=0)

    # Act
    data.loaders(64)
    calls_before_test = list(_FakeMNIST.calls)
    test_loader = data.test_loader(64)

    # Assert
    assert [call["train"] for call in calls_before_test] == [True, True]
    assert [call["train"] for call in _FakeMNIST.calls] == [True, True, False]
    assert len(test_loader.dataset) == 10_000


def test_mnist_data_loaders_are_cached(tmp_path, monkeypatch):
    """MNISTData should reuse DataLoaders for the same batch size."""
    # Arrange
    _FakeMNIST.calls.clear()
    monkeypatch.setattr(train_mnist.datasets, "MNIST", _FakeMNIST)
    data = train_mnist.MNISTData(tmp_path, num_workers=0)

    # Act
    first = data.loaders(2)
    second = data.loaders(2)

    # Assert
    assert first is second


def test_build_model_uses_grid_dimensions():
    """_build_model should apply channel and hidden-linear dimensions from the grid."""
    # Arrange
    hp = {"model_channels": 6, "hidden_linear_size": 12}

    # Act
    model = train_mnist._build_model(hp)

    # Assert
    assert model.conv1.out_channels == 6
    assert model.linear.out_features == 12


def test_minimal_mnist_net_outputs_class_logits():
    """MinimalMnistNet should produce one class-logit vector per input image."""
    # Arrange
    model = train_mnist.MinimalMnistNet()
    images = torch.zeros(2, 1, 28, 28)

    # Act
    output = model(images)

    # Assert
    assert output.shape == (2, train_mnist.NUM_CLASSES)
