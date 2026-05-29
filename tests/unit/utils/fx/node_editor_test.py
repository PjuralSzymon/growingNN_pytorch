"""NodeEditor.replace_submodule must resolve dotted parent paths via get_submodule."""

from pathlib import Path
import sys

import pytest
import torch.nn as nn
import torch.fx as fx

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from growingnn.utils.fx import NodeEditor


class _NestedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(4, 4)

    def forward(self, x):
        return self.inner(x)


class _ModelWithSequentialNested(nn.Module):
    """Mimics ResNet-style targets such as layer1.0.inner (parent path layer1.0)."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(_NestedBlock())

    def forward(self, x):
        return self.layer1(x)


class _ModelWithTopLevelChild(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 4)

    def forward(self, x):
        return self.l1(x)


class _ModelWithSingleLevelParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _NestedBlock()

    def forward(self, x):
        return self.block(x)


def test_dotted_parent_path_is_not_single_getattr_on_gm():
    """
    getattr(gm, "layer1.0") must fail: parent segments with a dot are not
    top-level attributes on the GraphModule.
    """
    # Arrange
    gm = fx.symbolic_trace(_ModelWithSequentialNested())
    parent_path = "layer1.0"

    # Act / Assert
    with pytest.raises(AttributeError):
        getattr(gm, parent_path)
    assert isinstance(gm.get_submodule(parent_path), nn.Module)


def test_replace_submodule_on_dotted_parent_path():
    """
    replace_submodule on layer1.0.inner must replace the leaf; broken parent
    lookup via getattr(gm, "layer1.0") would raise before add_module runs.
    """
    # Arrange
    gm = fx.symbolic_trace(_ModelWithSequentialNested())
    replacement = nn.Linear(4, 2)

    # Act
    NodeEditor.replace_submodule(gm, "layer1.0.inner", replacement)

    # Assert
    assert gm.get_submodule("layer1.0.inner") is replacement
    assert replacement.out_features == 2


def test_replace_submodule_on_single_level_parent_path():
    """
    replace_submodule still works when the parent is a single attribute (block).
    """
    # Arrange
    gm = fx.symbolic_trace(_ModelWithSingleLevelParent())
    replacement = nn.Linear(4, 2)

    # Act
    NodeEditor.replace_submodule(gm, "block.inner", replacement)

    # Assert
    assert gm.get_submodule("block.inner") is replacement
    assert replacement.out_features == 2


def test_replace_submodule_on_top_level_path():
    """
    replace_submodule with no dot in module_path replaces a direct child of gm.
    """
    # Arrange
    gm = fx.symbolic_trace(_ModelWithTopLevelChild())
    replacement = nn.Linear(4, 2)

    # Act
    NodeEditor.replace_submodule(gm, "l1", replacement)

    # Assert
    assert gm.get_submodule("l1") is replacement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
