"""Unit tests for ``growingnn.actions.utils.regularization_factory``."""

import torch.nn as nn

from growingnn.actions.utils.regularization_factory import RegularizationFactory


def test_create_dropout_returns_dropout_for_rank2_shape():
    """
    create_dropout should build nn.Dropout for 2D activation shapes.
    """
    # Arrange / Act
    layer = RegularizationFactory.create_dropout((2, 16), p=0.3)

    # Assert
    assert isinstance(layer, nn.Dropout)
    assert layer.p == 0.3


def test_create_dropout_returns_dropout2d_for_rank4_shape():
    """
    create_dropout should build nn.Dropout2d for 4D activation shapes.
    """
    # Arrange / Act
    layer = RegularizationFactory.create_dropout((2, 8, 7, 7), p=0.25)

    # Assert
    assert isinstance(layer, nn.Dropout2d)
    assert layer.p == 0.25


def test_create_dropout_returns_none_for_unsupported_rank():
    """
    create_dropout should return None when activation rank is not 2 or 4.
    """
    # Arrange / Act
    layer = RegularizationFactory.create_dropout((2, 8, 7), p=0.2)

    # Assert
    assert layer is None
