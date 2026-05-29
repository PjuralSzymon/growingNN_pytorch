"""Unit tests for ``growingnn.actions.utils.layer_Factory``."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.actions.utils.layer_Factory import (
    ConvFactory,
    Layer_Type,
    LinearFactory,
)


def test_create_zero_linear_has_zero_weights_and_bias():
    """
    create_zero_linear should zero-initialize weight and bias.
    """
    # Arrange / Act
    layer = LinearFactory.create_zero_linear(in_features=5, out_features=7)

    # Assert
    assert torch.all(layer.weight == 0)
    assert torch.all(layer.bias == 0)


def test_create_linear_dispatches_to_zero_random_and_eye():
    """
    create_linear should delegate to the matching factory for each Layer_Type.
    """
    # Arrange / Act
    zero = LinearFactory.create_linear(3, 4, Layer_Type.ZERO)
    random_layer = LinearFactory.create_linear(3, 4, Layer_Type.RANDOM)
    eye = LinearFactory.create_linear(4, 4, Layer_Type.EYE)

    # Assert
    assert torch.all(zero.weight == 0)
    assert not torch.all(random_layer.weight == 0)
    assert eye.weight.shape == (4, 4)


def test_create_linear_raises_for_unsupported_type():
    """
    create_linear should raise ValueError for unsupported enum values.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unsupported layer type"):
        LinearFactory.create_linear(2, 2, object())


def test_create_eye_linear_has_identity_like_square_weights():
    """
    create_eye_linear on square layers should place ones on the main diagonal.
    """
    # Arrange / Act
    layer = LinearFactory.create_eye_linear(4, 4)

    # Assert
    assert layer.weight.shape == (4, 4)
    assert torch.allclose(torch.diag(layer.weight), torch.ones(4))


def test_create_linear_with_rescaled_neurons_changes_out_features():
    """
    create_linear_with_rescaled_neurons should build a layer with fewer outputs.
    """
    # Arrange
    src = LinearFactory.create_random_linear(6, 8)

    # Act
    dst = LinearFactory.create_linear_with_rescaled_neurons(src, 4)

    # Assert
    assert dst.out_features == 4
    assert dst.in_features == 6


def test_create_linear_with_rescaled_connections_changes_in_features():
    """
    create_linear_with_rescaled_connections should build a layer with fewer inputs.
    """
    # Arrange
    src = LinearFactory.create_random_linear(8, 6)

    # Act
    dst = LinearFactory.create_linear_with_rescaled_connections(src, 4)

    # Assert
    assert dst.in_features == 4
    assert dst.out_features == 6


def test_create_zero_conv_has_zero_kernel_and_bias():
    """
    create_zero_conv should zero-initialize conv weights and bias.
    """
    # Arrange / Act
    conv = ConvFactory.create_zero_conv(3, 5, kernel_size=3, stride=1, padding=1)

    # Assert
    assert torch.all(conv.weight == 0)
    assert torch.all(conv.bias == 0)


def test_create_eye_conv_places_center_tap_on_channel_diagonal():
    """
    create_eye_conv should set centre kernel taps on matching in/out channels.
    """
    # Arrange / Act
    conv = ConvFactory.create_eye_conv(4, 4, kernel_size=3, stride=1, padding=1)

    # Assert
    assert conv.weight[0, 0, 1, 1].item() == 1.0
    assert conv.weight.detach().sum().item() == pytest.approx(4.0)


def test_create_conv_dispatches_zero_and_eye():
    """
    create_conv should build ZERO and EYE conv layers.
    """
    # Arrange / Act
    zero = ConvFactory.create_conv(3, 4, 3, 1, 1, Layer_Type.ZERO)
    eye = ConvFactory.create_conv(2, 2, 3, 1, 1, Layer_Type.EYE)

    # Assert
    assert torch.all(zero.weight == 0)
    assert eye.weight[0, 0, 1, 1] == 1.0


def test_create_conv_raises_for_random_type():
    """
    create_conv should not support Layer_Type.RANDOM.
    """
    # Arrange / Act / Assert
    with pytest.raises(ValueError, match="Unsupported layer type"):
        ConvFactory.create_conv(3, 3, 3, 1, 1, Layer_Type.RANDOM)


def test_create_zero_conv_before_linear_returns_sequential():
    """
    create_zero_conv_before_linear should wrap conv, pool, and flatten.
    """
    # Arrange / Act
    seq = ConvFactory.create_zero_conv_before_linear(3, 8, 3, 1, 1)

    # Assert
    assert isinstance(seq, nn.Sequential)
    assert len(seq) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
