"""Duplicate or remove shape-preserving residual regions in an FX graph.

Supported regions are closed, single-entry, single-output tensor regions
ending in an addition recognized by ``sum_nodes.is_sum_node``. Both operations
clone or bypass the entire residual region, including its skip path. This
supports identity-shape residual blocks such as ``x + F(x)`` and the project's
``residual_skip`` example.

Projection blocks that change shape are rejected. So are non-binary/opaque
merge forms, regions with side exits, and regions whose boundary shape cannot
be propagated. In particular, a standard ResNet-18's four stage-transition
blocks need shape-changing semantics and are not offered by this first
version; its four identity-shape blocks are candidates. Tracing must expose
the block operations as FX nodes for discovery to work.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.fx as fx

from growingnn.actions.action import Action
from growingnn.core.logger import logger
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser
from growingnn.utils.fx.node_analysis import ModuleResolver
from growingnn.utils.fx.sum_nodes import is_sum_node


def _ancestors(node: fx.Node) -> set[fx.Node]:
    result: set[fx.Node] = set()
    stack = list(node.all_input_nodes)
    while stack:
        current = stack.pop()
        if current not in result:
            result.add(current)
            stack.extend(current.all_input_nodes)
    return result


def _region_for_add(add: fx.Node, order: dict[fx.Node, int]) -> tuple[fx.Node, set[fx.Node]] | None:
    """Infer a closed one-entry region ending at a binary add."""
    if len(add.all_input_nodes) != 2:
        return None
    left, right = add.all_input_nodes
    # Include each branch value itself: for the common ``x + F(x)`` pattern,
    # the entry is the direct skip value ``x``.
    common = (_ancestors(left) | {left}) & (_ancestors(right) | {right})
    if not common:
        return None
    entry = max(common, key=order.__getitem__)
    ancestors = _ancestors(add)
    entry_ancestors = _ancestors(entry) | {entry}
    members = (ancestors - entry_ancestors) | {add}
    if not members or any(n.op in ("placeholder", "output") for n in members):
        return None
    # The entry is the only permitted external input; the add is the only exit.
    for node in members:
        if any(parent not in members and parent is not entry for parent in node.all_input_nodes):
            return None
        if any(user not in members and node is not add for user in node.users):
            return None
    return entry, members


def _discover(traced: TracedModel) -> list[tuple[fx.Node, fx.Node, set[fx.Node], str | None]]:
    gm = traced.gm
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    shapes = LayerShapeAnalyser
    # Populate ShapeProp metadata once. node_shape reads the metadata directly.
    shapes.collect_layer_shapes(gm, input_shape=traced.input_shape)
    found = []
    for add in gm.graph.nodes:
        if not is_sum_node(add):
            continue
        region = _region_for_add(add, order)
        if region is None:
            logger.debug("ResidualBlock skip %s: no closed single-entry region", add.name)
            continue
        entry, members = region
        in_shape = shapes.node_shape(entry)
        out_shape = shapes.node_shape(add)
        reason = None
        if in_shape is None or out_shape is None:
            reason = "boundary tensor shape unavailable"
        elif in_shape != out_shape:
            reason = f"shape-changing boundary {in_shape} -> {out_shape}"
        else:
            # A block may only be bypassed if its entry is valid for every user
            # outside the region (including the residual's downstream path).
            for user in add.users:
                user_shape = LayerShapeAnalyser.input_shape_for_layer(user) if user.op == "call_module" else None
                if user_shape is not None and user_shape != out_shape:
                    reason = f"incompatible external consumer {user.name}"
                    break
        found.append((entry, add, members, reason))
    return found


def discover_residual_regions(traced: TracedModel):
    """Public read-only discovery result for diagnostics and visualizers."""
    return _discover(traced)


def _reset_norm(module: nn.Module) -> None:
    with torch.no_grad():
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if weight is not None:
            weight.fill_(1.0)
        if bias is not None:
            bias.zero_()
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is not None:
            running_mean.zero_()
        if running_var is not None:
            running_var.fill_(1.0)
        batches = getattr(module, "num_batches_tracked", None)
        if batches is not None:
            batches.zero_()


def _initialize_duplicate_module(module: nn.Module) -> None:
    """Initialize copied Conv/Linear/norm modules toward channel-wise identity."""
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            module.weight.zero_()
            diagonal = min(module.in_features, module.out_features)
            indices = torch.arange(diagonal, device=module.weight.device)
            module.weight[indices, indices] = 1.0
            if module.bias is not None:
                module.bias.zero_()
        return
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
    if isinstance(module, conv_types):
        with torch.no_grad():
            module.weight.zero_()
            in_per_group = module.in_channels // module.groups
            out_per_group = module.out_channels // module.groups
            center = tuple((kernel - 1) // 2 for kernel in module.kernel_size)
            for group in range(module.groups):
                for channel in range(min(in_per_group, out_per_group)):
                    module.weight[(group * out_per_group + channel, channel, *center)] = 1.0
            if module.bias is not None:
                module.bias.zero_()
        return
    norm_types = (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )
    if isinstance(module, norm_types):
        _reset_norm(module)
        return
    children = list(module.children())
    if children:
        for child in children:
            _initialize_duplicate_module(child)
        return
    if any(True for _ in module.parameters(recurse=True)):
        raise TypeError(f"quasi-identity initialization is unsupported for module {type(module).__name__}")


class ResidualBlockAction(Action):
    """Grow or shrink a residual block; duplicated modules use quasi-identity weights.

    Action params: [operation, entry_node_name, exit_node_name].
    """

    def _execute(self, traced: TracedModel) -> None:
        operation, entry_name, exit_name = self.params
        gm = traced.gm
        by_name = {node.name: node for node in gm.graph.nodes}
        entry, exit_node = by_name[entry_name], by_name[exit_name]
        candidate = next((item for item in _discover(traced) if item[0] is entry and item[1] is exit_node), None)
        if candidate is None or candidate[3] is not None:
            raise ValueError("Residual block candidate is no longer valid")
        _, _, members, _ = candidate
        if operation == "remove":
            removed_targets = {str(node.target) for node in members if node.op == "call_module"}
            exit_node.replace_all_uses_with(entry)
            # Delete in reverse topological order after replacing external uses.
            for node in reversed(list(gm.graph.nodes)):
                if node in members and len(node.users) == 0:
                    gm.graph.erase_node(node)
            # Remove submodules only when no surviving call site refers to them.
            live_targets = {str(node.target) for node in gm.graph.nodes if node.op == "call_module"}
            for target in removed_targets - live_targets:
                try:
                    parent_path, _, leaf = target.rpartition(".")
                    parent = gm.get_submodule(parent_path) if parent_path else gm
                    delattr(parent, leaf)
                except AttributeError:
                    pass
        elif operation == "duplicate":
            env: dict[fx.Node, fx.Node] = {entry: exit_node}
            module_names: dict[str, str] = {}
            ordered_members = sorted(members, key=lambda node: list(gm.graph.nodes).index(node))
            original_users = list(exit_node.users)
            insertion_cursor = exit_node
            for node in ordered_members:
                with gm.graph.inserting_after(insertion_cursor):
                    if node is exit_node:
                        new_node = gm.graph.node_copy(node, lambda n: env.get(n, n))
                    elif node.op == "call_module":
                        old_name = str(node.target)
                        if old_name not in module_names:
                            base = old_name.rsplit(".", 1)[-1] + "_block"
                            new_name = ModuleResolver.unique_call_module_name(base, gm)
                            new_module = deepcopy(gm.get_submodule(old_name))
                            _initialize_duplicate_module(new_module)
                            gm.add_module(new_name, new_module)
                            module_names[old_name] = new_name
                        new_node = gm.graph.call_module(
                            module_names[old_name],
                            args=fx.map_arg(node.args, lambda n: env[n]),
                            kwargs=fx.map_arg(node.kwargs, lambda n: env[n]),
                        )
                    else:
                        new_node = gm.graph.node_copy(node, lambda n: env[n])
                env[node] = new_node
                insertion_cursor = new_node
            cloned_exit = env[exit_node]
            # Redirect only users that existed before cloning; the new block's
            # entry intentionally consumes the original exit value.
            for user in original_users:
                if user not in env.values() and user is not cloned_exit:
                    user.replace_input_with(exit_node, cloned_exit)
        else:
            raise ValueError(f"Unknown residual block operation: {operation}")
        gm.graph.lint()
        gm.recompile()

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        *,
        grow: bool = True,
        shrink: bool = True,
    ) -> list["ResidualBlockAction"]:
        actions: list[ResidualBlockAction] = []
        for entry, exit_node, members, reason in _discover(traced):
            if reason:
                logger.debug("ResidualBlock skip %s: %s", exit_node.name, reason)
                continue
            if grow:
                actions.append(ResidualBlockAction(["duplicate", entry.name, exit_node.name]))
            if shrink:
                actions.append(ResidualBlockAction(["remove", entry.name, exit_node.name]))
        return actions

    def __str__(self):
        return f"Residual Block Action: {self.params}"
