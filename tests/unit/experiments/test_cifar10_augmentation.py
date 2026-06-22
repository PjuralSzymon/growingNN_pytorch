"""Unit tests for CIFAR-10 augmentation factor in train_cifar10."""

import importlib.util
import sys
from pathlib import Path

import torchvision.transforms as transforms

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


def _transform_types(pipeline: transforms.Compose) -> tuple[type, ...]:
    return tuple(type(step) for step in pipeline.transforms)


def test_train_transform_factor_zero_uses_eval_pipeline_only():
    """
    augmentation_factor 0 should disable all stochastic train augmentations.
    """
    # Arrange
    module = _load_train_cifar10_module()

    # Act
    pipeline = module._train_transform(0.0)

    # Assert
    assert _transform_types(pipeline) == _transform_types(module._eval_transform())


def test_train_transform_factor_one_enables_full_augmentation_stack():
    """
    augmentation_factor 1 should use AutoAugment CIFAR-10 policy plus RandomErasing.
    """
    # Arrange
    module = _load_train_cifar10_module()

    # Act
    pipeline = module._train_transform(1.0)
    transform_types = _transform_types(pipeline)

    # Assert
    assert transform_types == (
        transforms.RandomCrop,
        transforms.RandomHorizontalFlip,
        transforms.AutoAugment,
        transforms.ToTensor,
        transforms.Normalize,
        transforms.RandomErasing,
    )


def test_train_transform_factor_scales_enabled_transform_count():
    """
    Higher augmentation_factor should enable progressively more transform types.
    """
    # Arrange
    module = _load_train_cifar10_module()

    # Act
    low = module._train_transform(0.1)
    high = module._train_transform(0.9)

    # Assert
    assert len(high.transforms) > len(low.transforms)
