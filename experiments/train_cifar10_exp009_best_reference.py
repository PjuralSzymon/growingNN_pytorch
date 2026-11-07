"""
Experiment 009: four image datasets for 20 generations.

MNIST starts from one convolution and one linear layer. CIFAR-10 starts from
two convolutions, two batch-norm layers, and two linear layers, with 128
channels and hidden size 256. CIFAR-100 keeps those convolutions and uses a
hidden layer of 512. Tiny ImageNet keeps those convolutions and uses a hidden layer of 768. Every action stays enabled. Twenty generations of
ten epochs. The cosine runs from 0.1 to 0.001 across those 200 epochs, and a
new generation does not restart it. Search is 8 seconds, 1 epoch, and 128
images. Seeds are 1 and 2.

    Datasets are MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet.
    Tiny ImageNet is read from experiments/data/tiny-imagenet-200.
    Pass dataset names to train only those, so two processes can run side by side.

    pipenv run python -u experiments/train_cifar10_exp009_best_reference.py
    pipenv run python -u experiments/train_cifar10_exp009_best_reference.py mnist cifar10
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import growingnn.simulation.simulation_algorithms.montecarlo_alg as montecarlo_alg
from experiments.train_cifar10 import Cifar10Data
from experiments.train_cifar10 import DATA_DIR as CIFAR_DIR
from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_mnist import DATA_DIR as MNIST_DIR
from growingnn.core.config import DATALOADER_NUM_WORKERS, RunningConfig
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.stoppers import AccuracyStopper
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all

RUNS_DIR = Path(__file__).resolve().parent / "output" / "train_cifar10" / "runs" / "exp009_small_20"
DATA_ROOT = Path(__file__).resolve().parent / "data"
SEEDS = (1, 2, 3)
DATASETS = ("mnist", "cifar10", "cifar100", "tiny_imagenet")
GENERATIONS = 20
EPOCHS = 10


class InitModelFactory:
    """MNIST uses one convolution. The color datasets use the four-layer CIFAR starter."""

    class MnistStarter(nn.Module):
        """1-channel MNIST stem. Eight filters, no BatchNorm, 152 parameters."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=False)
            self.linear2 = nn.Linear(8, 10, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.conv1(x))
            x = F.adaptive_avg_pool2d(x, 1)
            return self.linear2(x.flatten(1))

    class Cifar10Starter(nn.Module):
        """Two convolutions and two linear layers. 128 channels, hidden 256, 187,008 parameters."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.linear = nn.Linear(128, 256)
            self.linear2 = nn.Linear(256, 10, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.adaptive_avg_pool2d(x, 1)
            x = F.relu(self.linear(x.flatten(1)))
            return self.linear2(x)

    class Cifar100Starter(nn.Module):
        """CIFAR-10 convolutions with a hidden layer of 512 and a 100-way head. 268,672 parameters."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.linear = nn.Linear(128, 512)
            self.linear2 = nn.Linear(512, 100, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.adaptive_avg_pool2d(x, 1)
            x = F.relu(self.linear(x.flatten(1)))
            return self.linear2(x)

    class TinyImageNetStarter(nn.Module):
        """CIFAR-10 convolutions with a hidden layer of 768 and a 200-way head. 404,096 parameters."""

        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 128, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(128)
            self.conv2 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(128)
            self.linear = nn.Linear(128, 768)
            self.linear2 = nn.Linear(768, 200, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.adaptive_avg_pool2d(x, 1)
            x = F.relu(self.linear(x.flatten(1)))
            return self.linear2(x)

    @staticmethod
    def getmnist() -> MnistStarter:
        return InitModelFactory.MnistStarter()

    @staticmethod
    def getcifar10() -> Cifar10Starter:
        return InitModelFactory.Cifar10Starter()

    @staticmethod
    def getcifar100() -> Cifar100Starter:
        return InitModelFactory.Cifar100Starter()

    @staticmethod
    def gettinyimagenet() -> TinyImageNetStarter:
        return InitModelFactory.TinyImageNetStarter()


def initial_graph(dataset: str = "mnist") -> nn.Module:
    """Return the starter for one dataset."""
    if dataset == "mnist":
        return InitModelFactory.getmnist()
    if dataset == "cifar10":
        return InitModelFactory.getcifar10()
    if dataset == "cifar100":
        return InitModelFactory.getcifar100()
    if dataset == "tiny_imagenet":
        return InitModelFactory.gettinyimagenet()
    raise ValueError(f"unknown dataset {dataset}")


class DataFactory:
    """Loaders, splits, and the one method that opens each dataset."""

    class SplitData:
        """Augmented train, a held-out val slice, clean train, and the official eval split."""

        def __init__(self, root: Path, train_augmented, train_clean, evaluation, val_size: int) -> None:
            self.root = root
            self.val_size = val_size
            self._train_augmented = train_augmented
            self._train_clean = train_clean
            self._evaluation = evaluation
            self._datasets = None
            self._cache: dict[int, tuple] = {}

        def prepare(self) -> None:
            if self._datasets is not None:
                return
            clean = self._train_clean()
            augmented = self._train_augmented()
            val_size = min(self.val_size, len(clean) // 5)
            clean_train, val = random_split(
                clean,
                [len(clean) - val_size, val_size],
                generator=torch.Generator().manual_seed(0),
            )
            self._datasets = (
                DataFactory._with_targets(Subset(augmented, clean_train.indices), augmented, clean_train.indices),
                DataFactory._with_targets(val, clean, val.indices),
                DataFactory._with_targets(clean_train, clean, clean_train.indices),
            )

        def loaders(self, batch_size: int):
            self.prepare()
            if batch_size in self._cache:
                return self._cache[batch_size]
            train, val, clean = self._datasets
            pin_memory = torch.cuda.is_available()
            kwargs = {"batch_size": batch_size, "num_workers": DATALOADER_NUM_WORKERS, "pin_memory": pin_memory}
            loaders = (
                DataLoader(train, shuffle=True, **kwargs),
                DataLoader(val, **kwargs),
                DataLoader(clean, shuffle=False, **kwargs),
            )
            self._cache[batch_size] = loaders
            return loaders

        def test_loader(self, batch_size: int) -> DataLoader:
            self.prepare()
            return DataLoader(
                self._evaluation(),
                batch_size=batch_size,
                num_workers=DATALOADER_NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )

    class TinyImageNet(Dataset):
        """Stanford Tiny ImageNet. Train is class folders. Val uses val_annotations.txt when present."""

        def __init__(self, root: Path, split: str, transform) -> None:
            train_dir = root / "train"
            classes = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
            self.class_to_idx = {name: index for index, name in enumerate(classes)}
            self.transform = transform
            self.samples: list[Path] = []
            self.targets: list[int] = []
            if split == "train":
                for name in classes:
                    for path in DataFactory._image_files(train_dir / name):
                        self.samples.append(path)
                        self.targets.append(self.class_to_idx[name])
                return
            annotations = root / "val" / "val_annotations.txt"
            if annotations.is_file():
                for line in annotations.read_text(encoding="utf-8").splitlines():
                    filename, wnid, *_rest = line.split("\t")
                    self.samples.append(root / "val" / "images" / filename)
                    self.targets.append(self.class_to_idx[wnid])
                return
            for name in classes:
                for path in DataFactory._image_files(root / "val" / name):
                    self.samples.append(path)
                    self.targets.append(self.class_to_idx[name])

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            image = Image.open(self.samples[index]).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image, self.targets[index]

    @staticmethod
    def _with_targets(subset: Subset, base, indices) -> Subset:
        """Attach class labels so simulation sampling does not read every image."""
        subset.targets = torch.as_tensor(base.targets)[list(indices)]
        return subset

    @staticmethod
    def _tensor_transform(mean, std, *before):
        """Normalize after the listed image transforms."""
        return transforms.Compose([*before, transforms.ToTensor(), transforms.Normalize(mean, std)])

    @staticmethod
    def _image_files(folder: Path) -> list[Path]:
        """List images directly in a class folder, or in its official images subdirectory."""
        nested = folder / "images"
        source = nested if nested.is_dir() else folder
        return sorted(path for path in source.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})

    @staticmethod
    def getcifar10() -> Cifar10Data:
        data = Cifar10Data(CIFAR_DIR)
        data.prepare()
        return data

    @staticmethod
    def getcifar100() -> DataFactory.SplitData:
        root = DATA_ROOT / "cifar100"
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        aug = DataFactory._tensor_transform(mean, std, transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip())
        plain = DataFactory._tensor_transform(mean, std)
        return DataFactory.SplitData(
            root,
            lambda: datasets.CIFAR100(str(root), train=True, download=True, transform=aug),
            lambda: datasets.CIFAR100(str(root), train=True, download=True, transform=plain),
            lambda: datasets.CIFAR100(str(root), train=False, download=True, transform=plain),
            5_000,
        )

    @staticmethod
    def getmnist() -> DataFactory.SplitData:
        root = MNIST_DIR
        mean, std = (0.1307,), (0.3081,)
        aug = DataFactory._tensor_transform(mean, std, transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))
        plain = DataFactory._tensor_transform(mean, std)
        return DataFactory.SplitData(
            root,
            lambda: datasets.MNIST(str(root), train=True, download=True, transform=aug),
            lambda: datasets.MNIST(str(root), train=True, download=True, transform=plain),
            lambda: datasets.MNIST(str(root), train=False, download=True, transform=plain),
            10_000,
        )

    @staticmethod
    def download(dataset: str, root: Path | None = None) -> None:
        """Download one dataset into experiments/data when its folder is missing."""
        if dataset == "tiny_imagenet":
            DataFactory._download_tiny_imagenet(DATA_ROOT / "tiny-imagenet-200" if root is None else root)
            return
        raise ValueError(f"unknown dataset {dataset}")

    @staticmethod
    def _download_tiny_imagenet(root: Path) -> None:
        """Fetch the Zenodo Tiny ImageNet zip and unpack it at root."""
        if (root / "train").is_dir():
            return
        from torchvision.datasets.utils import download_and_extract_archive

        print(f"Downloading Tiny ImageNet to {root}", flush=True)
        root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root.parent) as staging:
            download_and_extract_archive(
                "https://zenodo.org/records/10720917/files/tiny-imagenet-200.zip?download=1",
                staging,
                filename="tiny-imagenet-200.zip",
                md5="90528d7ca1a48142e341f4ef8d21d0de",
                remove_finished=True,
            )
            extracted = Path(staging) / "tiny-imagenet-200"
            if not extracted.is_dir():
                raise FileNotFoundError(f"Tiny ImageNet archive did not contain {extracted.name}")
            root.mkdir(parents=True, exist_ok=True)
            for child in extracted.iterdir():
                shutil.move(str(child), root / child.name)

    @staticmethod
    def gettinyimagenet(root: Path | None = None) -> DataFactory.SplitData:
        root = DATA_ROOT / "tiny-imagenet-200" if root is None else root
        if not (root / "train").is_dir():
            DataFactory.download("tiny_imagenet", root)
        if not (root / "train").is_dir():
            raise FileNotFoundError(f"Tiny ImageNet is required at {root}")
        mean, std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
        aug = DataFactory._tensor_transform(mean, std, transforms.RandomCrop(64, padding=4), transforms.RandomHorizontalFlip())
        plain = DataFactory._tensor_transform(mean, std)
        return DataFactory.SplitData(
            root,
            lambda: DataFactory.TinyImageNet(root, "train", aug),
            lambda: DataFactory.TinyImageNet(root, "train", plain),
            lambda: DataFactory.TinyImageNet(root, "val", plain),
            10_000,
        )


def get_datasets(dataset: str):
    """Open the loader named by dataset. Each name calls one DataFactory method."""
    if dataset == "mnist":
        return DataFactory.getmnist()
    if dataset == "cifar10":
        return DataFactory.getcifar10()
    if dataset == "cifar100":
        return DataFactory.getcifar100()
    if dataset == "tiny_imagenet":
        return DataFactory.gettinyimagenet()
    raise ValueError(f"unknown dataset {dataset}")


def train_seed(data, seed: int, out_dir: Path, dataset: str = "cifar10") -> None:
    """Train one seed, claiming its folder first so a second process skips a started seed."""
    history_path = out_dir / "history.pt"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_dir.mkdir(exist_ok=False)
    except FileExistsError:
        state = "history exists" if history_path.is_file() else "already started"
        print(f"SKIP {dataset} seed {seed} {state} at {out_dir}", flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(seed)
    model = initial_graph(dataset)
    train_loader, val_loader, clean_loader = data.loaders(128)
    train_loader = with_cutout(train_loader)
    test_loader = data.test_loader(128)
    sim_train, sim_val = sample_loaders(clean_loader, val_loader, 128, seed=seed)
    config = RunningConfig(
        generations=GENERATIONS,
        epochs=EPOCHS,
        lr_scheduler=CosineDecayScheduler(total_epochs=GENERATIONS * EPOCHS),
        stopper=AccuracyStopper(target_accuracy=0.999),
        simulation_alg=montecarlo_alg,
        simulation_scheduler=AlwaysSimulationScheduler(8.0, 1),
        simulation_score=SimulationScore(weight_acc=1.0, weight_countW=0.0),
        simulation_set_size=128,
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        device=device,
        momentum=0.9,
        weight_decay=0.0001,
        max_grad_norm=1.0,
        seq_conv_insert_before_global_pool=True,
    )
    print(
        f"EXP009 dataset {dataset} seed {seed} device {device} model {type(model).__name__} "
        f"generations {GENERATIONS} epochs {EPOCHS} lr CosineDecay 0.1 to 0.001 no restart "
        f"smooth 0.1 cutout True wd 0.0001 clip 1.0 momentum 0.9 "
        f"actions all sim 8s 1 epoch set 128",
        flush=True,
    )
    _grown, summary = train_generations(
        model,
        train_loader,
        val_loader,
        config,
        sim_train_loader=sim_train,
        sim_val_loader=sim_val,
        test_loader=test_loader,
    )
    torch.save({"seed": seed, "dataset": dataset, **{key: list(summary[key]) for key in summary}}, history_path)
    print(f"TEST {dataset} seed {seed} {[round(float(value), 4) for value in summary['test_acc']]}", flush=True)


def selected_datasets(names: list[str]) -> tuple[str, ...]:
    """Return the named datasets, or every dataset when no names are given."""
    if not names:
        return DATASETS
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset {unknown[0]}")
    return tuple(names)


def main() -> None:
    """Train MNIST, CIFAR-10, CIFAR-100, and Tiny ImageNet for 20 generations on seeds 1 and 2."""
    for dataset in selected_datasets(sys.argv[1:]):
        data = get_datasets(dataset)
        print(f"Write target: {RUNS_DIR / dataset}", flush=True)
        for seed in SEEDS:
            train_seed(data, seed, RUNS_DIR / dataset / f"seed_{seed}", dataset)


if __name__ == "__main__":
    main()
