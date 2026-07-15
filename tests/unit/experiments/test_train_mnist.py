"""Unit tests for dataset-source organization in train_mnist."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from experiments import train_mnist


def test_create_dataset_configurations_covers_selected_datasets():
    """
    Runtime dataset configuration should contain every dataset selected by the grid.
    """
    # Arrange
    expected_datasets = set(train_mnist.DATASET_ORDER)

    # Act
    configurations = train_mnist._create_dataset_configurations()

    # Assert
    assert set(configurations) == expected_datasets


def test_torchvision_source_detects_downloaded_raw_data(tmp_path):
    """
    TorchvisionDatasetSource should detect a dataset with files in its raw directory.
    """
    # Arrange
    raw_dir = tmp_path / "MNIST" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "data.gz").write_bytes(b"cached")

    # Act
    downloaded = train_mnist.TorchvisionDatasetSource.is_downloaded(tmp_path, "MNIST")

    # Assert
    assert downloaded is True


def test_medmnist_source_detects_downloaded_npz_data(tmp_path):
    """
    MedMnistDatasetSource should detect the dataset's NPZ cache file.
    """
    # Arrange
    (tmp_path / "breastmnist.npz").write_bytes(b"cached")

    # Act
    downloaded = train_mnist.MedMnistDatasetSource.is_downloaded(
        tmp_path, "breastmnist"
    )

    # Assert
    assert downloaded is True


def test_torchvision_source_creates_training_dataset_with_split(tmp_path):
    """
    TorchvisionDatasetSource should create a training builder with download and split settings.
    """
    # Arrange
    calls = []

    def dataset_factory(**kwargs):
        calls.append(kwargs)
        return kwargs

    spec = train_mnist.TorchvisionDatasetSource.create_spec(
        "em",
        dataset_factory,
        47,
        (0.1,),
        (0.2,),
        dataset_name="EMNIST",
        emnist_split="balanced",
    )

    # Act
    result = spec.build_train(tmp_path, True, False)

    # Assert
    assert result == calls[0]
    assert calls[0]["train"] is True
    assert calls[0]["download"] is True
    assert calls[0]["split"] == "balanced"


def test_medmnist_source_converts_array_label_to_scalar(tmp_path):
    """
    MedMnistDatasetSource should wrap MedMNIST datasets so labels are scalar integers.
    """
    # Arrange
    class FakeMedMnist(Dataset):
        def __init__(self, **_kwargs):
            pass

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int):
            return torch.zeros(1, 28, 28), [1]

    spec = train_mnist.MedMnistDatasetSource.create_spec(
        "breastm",
        FakeMedMnist,
        2,
        1,
        (0.5,),
        (0.5,),
        dataset_name="breastmnist",
    )

    # Act
    _, label = spec.build_train(Path(tmp_path), False, False)[0]

    # Assert
    assert label == 1
