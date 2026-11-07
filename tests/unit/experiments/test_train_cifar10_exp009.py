"""Unit tests for the Experiment 009 reference recipe."""

from pathlib import Path

import pytest
import torch.nn as nn

from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler
from experiments.train_cifar10_exp009_best_reference import (
    DataFactory,
    InitModelFactory,
    get_datasets,
    initial_graph,
    selected_datasets,
    train_seed,
)
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler


def test_initial_graph_uses_the_high_result_cifar10_starter():
    """
    initial_graph should build the CIFAR-10 starter with two 128-channel convolutions and 187,008 parameters.
    """

    # Arrange

    # Act
    model = initial_graph("cifar10")

    # Assert
    assert type(model) is InitModelFactory.Cifar10Starter
    assert model.conv1.out_channels == 128
    assert model.conv2.out_channels == 128
    assert model.linear.out_features == 256
    assert model.linear2.out_features == 10
    assert sum(parameter.numel() for parameter in model.parameters()) == 187_008


def test_initial_graph_keeps_mnist_small_and_widens_only_the_color_heads():
    """
    initial_graph should keep the 152-parameter MNIST stem, use a CIFAR-100 hidden layer of 512, and a Tiny ImageNet hidden layer of 768.
    """

    # Arrange

    # Act
    mnist = initial_graph("mnist")
    cifar100 = initial_graph("cifar100")
    tiny = initial_graph("tiny_imagenet")

    # Assert
    assert isinstance(mnist, InitModelFactory.MnistStarter)
    assert mnist.conv1.in_channels == 1
    assert not hasattr(mnist, "conv2")
    assert sum(parameter.numel() for parameter in mnist.parameters()) == 152
    assert type(cifar100) is InitModelFactory.Cifar100Starter
    assert cifar100.conv1.out_channels == 128
    assert cifar100.linear.out_features == 512
    assert cifar100.linear2.out_features == 100
    assert sum(parameter.numel() for parameter in cifar100.parameters()) == 268_672
    assert type(tiny) is InitModelFactory.TinyImageNetStarter
    assert tiny.conv1.out_channels == 128
    assert tiny.linear.out_features == 768
    assert tiny.linear2.out_features == 200
    assert sum(parameter.numel() for parameter in tiny.parameters()) == 404_096


def test_tiny_imagenet_reads_train_class_folders_and_val_annotations(tmp_path):
    """
    TinyImageNet should read train/class/images and the official val annotation file.
    """

    # Arrange
    train_images = tmp_path / "train" / "n00000001" / "images"
    val_images = tmp_path / "val" / "images"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "a.JPEG").write_bytes(b"")
    (val_images / "b.JPEG").write_bytes(b"")
    (tmp_path / "val" / "val_annotations.txt").write_text("b.JPEG\tn00000001\t0\t0\t1\t1\n", encoding="utf-8")

    # Act
    train = DataFactory.TinyImageNet(tmp_path, "train", transform=None)
    val = DataFactory.TinyImageNet(tmp_path, "val", transform=None)

    # Assert
    assert len(train) == 1
    assert train.targets == [0]
    assert len(val) == 1
    assert val.targets == [0]


def test_download_unpacks_tiny_imagenet_into_the_dataset_root(tmp_path, monkeypatch):
    """
    download should unpack the Tiny ImageNet archive so root/train exists.
    """

    # Arrange
    def fake_extract(_url, download_root, filename, md5, remove_finished):
        assert filename == "tiny-imagenet-200.zip"
        assert md5
        assert remove_finished is True
        (Path(download_root) / "tiny-imagenet-200" / "train").mkdir(parents=True)

    monkeypatch.setattr("torchvision.datasets.utils.download_and_extract_archive", fake_extract)
    root = tmp_path / "tiny-imagenet-200"

    # Act
    DataFactory.download("tiny_imagenet", root)

    # Assert
    assert (root / "train").is_dir()


def test_get_datasets_calls_the_matching_factory_method(monkeypatch):
    """
    get_datasets should call DataFactory.getcifar100 when the name is cifar100.
    """

    # Arrange
    monkeypatch.setattr(DataFactory, "getcifar100", staticmethod(lambda: "cifar100-data"))

    # Act
    result = get_datasets("cifar100")

    # Assert
    assert result == "cifar100-data"


def test_factory_methods_build_the_held_out_split(tmp_path):
    """
    Each DataFactory method should return a SplitData with that dataset's validation size.
    """

    # Arrange
    tiny_root = tmp_path / "tiny"
    (tiny_root / "train").mkdir(parents=True)

    # Act
    cifar100 = DataFactory.getcifar100()
    mnist = DataFactory.getmnist()
    tiny = DataFactory.gettinyimagenet(tiny_root)

    # Assert
    assert isinstance(cifar100, DataFactory.SplitData) and cifar100.val_size == 5_000
    assert isinstance(mnist, DataFactory.SplitData) and mnist.val_size == 10_000
    assert isinstance(tiny, DataFactory.SplitData) and tiny.val_size == 10_000


def test_selected_datasets_keeps_two_named_datasets_and_rejects_the_rest():
    """
    selected_datasets should return the requested pair and reject a dataset that is not in the run.
    """

    # Arrange

    # Act
    chosen = selected_datasets(["mnist", "cifar10"])

    # Assert
    assert chosen == ("mnist", "cifar10")
    with pytest.raises(ValueError, match="unknown dataset"):
        selected_datasets(["imagenet"])


def test_get_datasets_rejects_an_unknown_dataset():
    """
    get_datasets should raise ValueError when the dataset name is not one of the four.
    """

    # Arrange

    # Act / Assert
    with pytest.raises(ValueError, match="unknown dataset"):
        get_datasets("fmnist")


def test_train_seed_keeps_the_cosine_across_generations_and_all_actions(tmp_path, monkeypatch):
    """
    train_seed should pass 20 generations of 10 epochs with every action enabled and the cosine unrestarted.
    """

    # Arrange
    captured = {}

    def _loaders(_batch_size):
        return object(), object(), object()

    out_dir = tmp_path / "seed_110"

    def _train(*_args, **kwargs):
        captured["config"] = _args[3]
        captured["folder_claimed"] = out_dir.is_dir()
        return _args[0], {"test_acc": [0.91]}

    monkeypatch.setattr(
        "experiments.train_cifar10_exp009_best_reference.train_generations",
        _train,
    )
    monkeypatch.setattr(
        "experiments.train_cifar10_exp009_best_reference.sample_loaders",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "experiments.train_cifar10_exp009_best_reference.with_cutout",
        lambda loader: loader,
    )

    class _Data:
        loaders = staticmethod(_loaders)
        test_loader = staticmethod(lambda _batch_size: object())

    # Act
    train_seed(_Data(), 110, out_dir)
    config = captured["config"]
    before = config.lr_scheduler.global_epoch
    config.lr_scheduler.structure_changed()

    # Assert
    assert config.generations == 20
    assert config.epochs == 10
    assert isinstance(config.lr_scheduler, CosineDecayScheduler)
    assert config.lr_scheduler.total_epochs == 200
    assert config.lr_scheduler.global_epoch == before
    assert config.momentum == 0.9
    assert config.weight_decay == 0.0001
    assert config.max_grad_norm == 1.0
    assert isinstance(config.criterion, nn.CrossEntropyLoss)
    assert config.criterion.label_smoothing == 0.1
    assert config.seq_conv_insert_before_global_pool is True
    assert isinstance(config.simulation_scheduler, AlwaysSimulationScheduler)
    assert config.simulation_scheduler.simulation_time == 8.0
    assert config.simulation_scheduler.simulation_epochs == 1
    assert config.simulation_set_size == 128
    assert config.ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER is True
    assert config.ACTIONS_ENABLE_ADD_RES_CONV_LAYER is True
    assert config.ACTIONS_ENABLE_ADD_SEQ_LAYER is True
    assert config.ACTIONS_ENABLE_DEL_LAYER is True
    assert config.ACTIONS_ENABLE_ADD_NEURONS_15 is True
    assert captured["folder_claimed"] is True
    assert (out_dir / "history.pt").is_file()


def test_train_seed_skips_a_seed_whose_folder_already_exists(tmp_path, monkeypatch):
    """
    train_seed should skip a seed when its output folder already exists, even without history.pt.
    """

    # Arrange
    out_dir = tmp_path / "seed_2"
    out_dir.mkdir()
    called = []
    monkeypatch.setattr(
        "experiments.train_cifar10_exp009_best_reference.train_generations",
        lambda *_args, **_kwargs: called.append(True),
    )

    # Act
    train_seed(object(), 2, out_dir, "mnist")

    # Assert
    assert called == []
    assert list(out_dir.iterdir()) == []
