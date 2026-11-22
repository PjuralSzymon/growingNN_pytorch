"""Insert a normalization layer after an editable layer, before its activation."""

from __future__ import annotations

import torch.fx as fx
from torch import nn

from growingnn.actions.action import Action
from growingnn.core import config
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx import ModuleResolver
from growingnn.utils.fx.graph_editor import _align_new_module_to_graph
from growingnn.utils.fx.sum_nodes import _flatten_terms, is_sum_node

NORMALIZATION_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm,
                       nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)


def normalization_for(module: nn.Module, shape: tuple[int, ...]) -> tuple[str, int] | None:
    """Normalization kind and width that fit the layer's output, or None.

    Convolutions get the BatchNorm of their dimension over channels. A linear layer with a
    (batch, features) output gets BatchNorm1d; with a longer (batch, ..., features) output,
    such as a sequence, LayerNorm over the features.
    """
    if isinstance(module, nn.Conv1d) and len(shape) == 3:
        return 'BatchNorm1d', shape[1]
    if isinstance(module, nn.Conv2d) and len(shape) == 4:
        return 'BatchNorm2d', shape[1]
    if isinstance(module, nn.Conv3d) and len(shape) == 5:
        return 'BatchNorm3d', shape[1]
    if isinstance(module, nn.Linear):
        if len(shape) == 2:
            return 'BatchNorm1d', shape[1]
        if len(shape) > 2:
            return 'LayerNorm', shape[-1]
    return None


def build_normalization(kind: str, width: int) -> nn.Module:
    """A new normalization layer with PyTorch's default initialization."""
    return {'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'BatchNorm3d': nn.BatchNorm3d,
            'LayerNorm': nn.LayerNorm}[kind](width)


def output_tail(node: fx.Node) -> fx.Node | None:
    """The node that carries the layer's full output: the layer, or the residual sum into it.

    Residual branches are summed into a layer's raw output, so a normalization of that layer
    must come after the sum. Returns None for a layer that only feeds sums as a branch.
    """
    trunk_sums = [user for user in node.users if is_sum_node(user) and _flatten_terms(user)[0] is node]
    if trunk_sums:
        return trunk_sums[0]
    if node.users and all(is_sum_node(user) for user in node.users):
        return None
    return node


def _is_normalization(node: fx.Node, gm: fx.GraphModule) -> bool:
    return node.op == 'call_module' and isinstance(ModuleResolver.get_layer_module(node, gm), NORMALIZATION_TYPES)


class AddNormalization(Action):
    """params: [layer name, normalization kind, width, new module name].

    The new layer keeps PyTorch's default initialization (BatchNorm: weight 1, bias 0, running
    mean 0, running variance 1). A BatchNorm leaves the evaluation output unchanged; in training
    mode it normalizes with batch statistics from the first step, so the output changes. A
    LayerNorm normalizes each sample in both modes, so it changes the output in both.
    """

    def _execute(self, traced: TracedModel) -> None:
        layer_name, kind, width, name = self.params
        gm = traced.gm
        layer = ModuleResolver.find_call_module(list(gm.graph.nodes), layer_name)
        tail = output_tail(layer)
        if tail is None:
            raise ValueError(f'{layer_name!r} only feeds residual sums; there is no output to normalize.')
        gm.add_module(name, _align_new_module_to_graph(gm, build_normalization(kind, width)))
        with gm.graph.inserting_after(tail):
            norm = gm.graph.call_module(name, args=(tail,))
        tail.replace_all_uses_with(norm, delete_user_cb=lambda user: user is not norm)
        gm.graph.lint()
        gm.recompile()

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(traced: TracedModel) -> list[Action]:
        """Offer a normalization after every editable layer that has none and does not feed the output."""
        gm = traced.gm
        out_shapes, _ = traced.shapes()
        actions: list[Action] = []
        for node in gm.graph.nodes:
            if node.op != 'call_module':
                continue
            module = ModuleResolver.get_layer_module(node, gm)
            if not isinstance(module, tuple(config.EDITABLE_MODULES)):
                continue
            tail = output_tail(node)
            if tail is None or any(user.op == 'output' or _is_normalization(user, gm) for user in tail.users):
                continue
            shape = out_shapes.get(str(node.target))
            norm = normalization_for(module, shape) if shape is not None else None
            if norm is None:
                continue
            name = ModuleResolver.unique_call_module_name('norm', gm)
            actions.append(AddNormalization([str(node.target), *norm, name]))
        return actions

    def __str__(self):
        return f' ( Add Normalization Action: {self.params} ) '
