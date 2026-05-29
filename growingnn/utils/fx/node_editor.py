"""Per-node edits: submodule replacement and node input rewiring."""

from __future__ import annotations

import torch.fx as fx


class NodeEditor:
    """Primitives for modifying individual submodules and node connections."""

    @staticmethod
    def replace_submodule(gm, module_path: str, new_module):
        """Replace the submodule at *module_path* (e.g. ``'layer1.0.conv1'``) with *new_module*."""
        parent, _, leaf = module_path.rpartition(".")
        parent_mod = gm if not parent else gm.get_submodule(parent)
        parent_mod.add_module(leaf, new_module)

    @staticmethod
    def swap_node_input(node: fx.Node, old: fx.Node, new: fx.Node):
        """Swap one input of *node* from *old* to *new*."""
        if old in node.args:
            node.args = tuple(new if a is old else a for a in node.args)
        if node.kwargs and old in node.kwargs.values():
            node.kwargs = {k: (new if v is old else v) for k, v in node.kwargs.items()}
