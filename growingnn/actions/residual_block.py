"""Duplicate or shrink shape-preserving residual blocks in an FX graph.

Supported regions are closed, single-entry, single-output tensor regions
ending in an addition recognized by ``sum_nodes.is_sum_node``. This supports
identity-shape residual blocks such as ``relu(x + F(x))`` and the project's
``residual_skip`` example. A block stays discoverable when another action
(for example ``AddResConvLayer``) wired a side input from an earlier layer
into its body.

Operations
----------
``duplicate``
    Insert a copy of the block right after it. When the block's add is
    followed by an activation (``relu(x + F(x))``), the copy goes after that
    activation and includes its own, so the two blocks are separated by a
    nonlinearity. Copied modules use quasi-identity weights (identity
    conv/linear, reset norm). Side inputs are not copied: the copy computes
    only ``y + F'(y)``.
``remove``
    Bypass the whole block (its exit is replaced by its entry). Side
    branches feeding only the block are deleted with it. Not offered for the
    last residual block in the graph.
``merge``
    Offered instead of ``remove`` when two identical blocks are adjacent:
    the second block is folded into the first (parameters averaged, last
    branch layer summed) and then removed.

Projection blocks that change shape are rejected, as are regions with side
exits or whose boundary shape cannot be propagated.
"""

from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fx as fx

from growingnn.actions.action import Action
from growingnn.core.logger import logger
from growingnn.core.traced_model import TracedModel
from growingnn.utils.fx.graph_analysis import LayerShapeAnalyser
from growingnn.utils.fx.node_analysis import ModuleResolver
from growingnn.utils.fx.sum_nodes import is_sum_node

_ACTIVATION_FUNCTIONS = {F.relu, torch.relu, F.gelu, F.silu, F.leaky_relu, F.elu, torch.tanh, torch.sigmoid}
_ACTIVATION_METHODS = {"relu", "gelu", "tanh", "sigmoid"}
_ACTIVATION_MODULES = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.Sigmoid)
# Parameter-free ops that are positively homogeneous: scaling their input scales their output.
_HOMOGENEOUS_FUNCTIONS = {F.relu, torch.relu, F.dropout, F.dropout2d, F.max_pool2d, F.avg_pool2d}
_HOMOGENEOUS_MODULES = (nn.ReLU, nn.Dropout, nn.Dropout2d, nn.Identity, nn.MaxPool2d, nn.AvgPool2d)
_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.SyncBatchNorm, nn.GroupNorm, nn.LayerNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)
_WEIGHT_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)


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
    """Infer a closed one-entry region ending at a binary add.

    The body may also read nodes that lie before the entry (a side input from
    an earlier layer into the block). Those nodes stay outside the region;
    everything else must enter through the entry.
    """
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
    # The entry and earlier layers are the only permitted external inputs;
    # the add is the only exit.
    for node in members:
        if any(parent not in members and parent not in entry_ancestors for parent in node.all_input_nodes):
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
    if isinstance(module, _NORM_TYPES):
        _reset_norm(module)
        return
    children = list(module.children())
    if children:
        for child in children:
            _initialize_duplicate_module(child)
        return
    if any(True for _ in module.parameters(recurse=True)):
        raise TypeError(f"quasi-identity initialization is unsupported for module {type(module).__name__}")


def _is_activation(gm: fx.GraphModule, node: fx.Node) -> bool:
    if len(node.all_input_nodes) != 1:
        return False
    if node.op == "call_function":
        return node.target in _ACTIVATION_FUNCTIONS
    if node.op == "call_method":
        return node.target in _ACTIVATION_METHODS
    if node.op == "call_module":
        return isinstance(gm.get_submodule(str(node.target)), _ACTIVATION_MODULES)
    return False


def _trailing_activation(gm: fx.GraphModule, add: fx.Node) -> fx.Node | None:
    """Return the activation applied directly (and only) to the block's add."""
    users = list(add.users)
    if len(users) == 1 and _is_activation(gm, users[0]):
        return users[0]
    return None


def _branch_tail_modules(gm: fx.GraphModule, add: fx.Node, skip: fx.Node) -> list[tuple[fx.Node, nn.Module]]:
    """Find the parameterized modules whose output feeds ``add`` on the branch side.

    Walk back from the non-skip input of ``add`` through positively
    homogeneous ops (ReLU, dropout, pooling) and sums. Merge sums these
    modules' parameters, because both blocks' branch outputs were added.
    Unsupported ops end the walk and are logged.
    """
    branch_inputs = [n for n in add.all_input_nodes if n is not skip]
    found: list[tuple[fx.Node, nn.Module]] = []
    seen: set[fx.Node] = set()
    stack = list(branch_inputs)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        if is_sum_node(node):
            stack.extend(node.all_input_nodes)
            continue
        if node.op == "call_module":
            module = gm.get_submodule(str(node.target))
            if isinstance(module, _NORM_TYPES + _WEIGHT_TYPES):
                found.append((node, module))
                continue
            if isinstance(module, _HOMOGENEOUS_MODULES):
                stack.extend(node.all_input_nodes)
                continue
        if node.op == "call_function" and node.target in _HOMOGENEOUS_FUNCTIONS:
            stack.extend(node.all_input_nodes[:1])
            continue
        logger.debug("ResidualBlock branch tail search stops at %s", node.name)
    return found


def _module_signature(gm: fx.GraphModule, node: fx.Node) -> tuple:
    if node.op == "call_module":
        module = gm.get_submodule(str(node.target))
        shapes = tuple(tuple(p.shape) for p in module.parameters(recurse=True))
        return ("module", type(module).__name__, shapes)
    return (node.op, str(node.target), len(node.all_input_nodes))


def _ordered(gm: fx.GraphModule, nodes: set[fx.Node]) -> list[fx.Node]:
    return [node for node in gm.graph.nodes if node in nodes]


def _signature(gm: fx.GraphModule, members: set[fx.Node]) -> tuple:
    return tuple(_module_signature(gm, node) for node in _ordered(gm, members))


def _remove_region(gm: fx.GraphModule, entry: fx.Node, exit_node: fx.Node, members: set[fx.Node]) -> None:
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


def _adjacent_pairs(gm: fx.GraphModule, blocks: list[tuple[fx.Node, fx.Node, set[fx.Node]]]):
    """Yield (first, second) blocks where second starts right after first and both match."""
    for first in blocks:
        entry_a, exit_a, members_a = first
        activation = _trailing_activation(gm, exit_a)
        for second in blocks:
            entry_b, exit_b, members_b = second
            if second is first:
                continue
            if entry_b is not exit_a and entry_b is not activation:
                continue
            if _signature(gm, members_a) != _signature(gm, members_b):
                continue
            yield first, second


class ResidualBlockAction(Action):
    """Grow or shrink a residual block.

    Action params:
        ``["duplicate", entry, exit]``, ``["remove", entry, exit]``, or
        ``["merge", entry_a, exit_a, entry_b, exit_b]`` (block b folds into block a).
    """

    def _valid_block(self, traced: TracedModel, entry_name: str, exit_name: str):
        by_name = {node.name: node for node in traced.gm.graph.nodes}
        entry, exit_node = by_name[entry_name], by_name[exit_name]
        candidate = next(
            (item for item in _discover(traced) if item[0] is entry and item[1] is exit_node),
            None,
        )
        if candidate is None or candidate[3] is not None:
            raise ValueError("Residual block candidate is no longer valid")
        return candidate[0], candidate[1], candidate[2]

    def _execute(self, traced: TracedModel) -> None:
        operation = self.params[0]
        gm = traced.gm
        if operation == "remove":
            entry, exit_node, members = self._valid_block(traced, self.params[1], self.params[2])
            _remove_region(gm, entry, exit_node, members)
        elif operation == "duplicate":
            entry, exit_node, members = self._valid_block(traced, self.params[1], self.params[2])
            self._duplicate(gm, entry, exit_node, members)
        elif operation == "merge":
            first = self._valid_block(traced, self.params[1], self.params[2])
            second = self._valid_block(traced, self.params[3], self.params[4])
            self._merge(gm, first, second)
        else:
            raise ValueError(f"Unknown residual block operation: {operation}")
        gm.graph.lint()
        gm.recompile()

    def _duplicate(self, gm: fx.GraphModule, entry: fx.Node, add: fx.Node, members: set[fx.Node]) -> None:
        exit_node = add
        copy_members = set(members)
        activation = _trailing_activation(gm, add)
        if activation is not None:
            exit_node = activation
            copy_members.add(activation)
        # Side nodes read only earlier layers. They are left out of the copy, so the
        # copy computes y + F'(y) and does not add the side input a second time.
        entry_side = {entry}
        for node in _ordered(gm, members):
            if any(parent in entry_side for parent in node.all_input_nodes):
                entry_side.add(node)
        side_nodes = members - entry_side
        to_copy = copy_members - side_nodes

        env: dict[fx.Node, fx.Node] = {entry: exit_node}

        def lookup(n: fx.Node) -> fx.Node:
            if n in env:
                return env[n]
            if n in to_copy:
                raise KeyError(f"copy of {n.name} used before it was created")
            return n  # external node

        module_names: dict[str, str] = {}
        original_users = list(exit_node.users)
        insertion_cursor = exit_node
        for node in _ordered(gm, to_copy):
            if is_sum_node(node):
                terms = [lookup(term) for term in node.all_input_nodes if term not in side_nodes]
                if len(terms) == 1:
                    # Only the side input was summed here; the copy passes the value through.
                    env[node] = terms[0]
                    continue
                with gm.graph.inserting_after(insertion_cursor):
                    new_node = gm.graph.call_function(node.target, args=tuple(terms))
            else:
                with gm.graph.inserting_after(insertion_cursor):
                    if node.op == "call_module" and not _is_activation(gm, node):
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
                            args=fx.map_arg(node.args, lookup),
                            kwargs=fx.map_arg(node.kwargs, lookup),
                        )
                    else:
                        new_node = gm.graph.node_copy(node, lookup)
            env[node] = new_node
            insertion_cursor = new_node
        cloned_exit = env[exit_node]
        # Redirect only users that existed before cloning; the new block's
        # entry intentionally consumes the original exit value.
        for user in original_users:
            if user not in env.values() and user is not cloned_exit:
                user.replace_input_with(exit_node, cloned_exit)

    def _merge(self, gm: fx.GraphModule, first, second) -> None:
        entry_a, add_a, members_a = first
        entry_b, add_b, members_b = second
        ordered_a = _ordered(gm, members_a)
        ordered_b = _ordered(gm, members_b)
        skip_a = entry_a
        tail_targets = {str(node.target) for node, _ in _branch_tail_modules(gm, add_a, skip_a)}
        for node_a, node_b in zip(ordered_a, ordered_b):
            if node_a.op != "call_module" or node_b.op != "call_module" or node_a.target == node_b.target:
                continue
            module_a = gm.get_submodule(str(node_a.target))
            module_b = gm.get_submodule(str(node_b.target))
            is_tail = str(node_a.target) in tail_targets
            with torch.no_grad():
                for (_, pa), (_, pb) in zip(module_a.named_parameters(), module_b.named_parameters()):
                    if is_tail:
                        pa.add_(pb)  # both branches added their outputs; keep the sum
                    else:
                        pa.add_(pb).mul_(0.5)
                for (name, ba), (_, bb) in zip(module_a.named_buffers(), module_b.named_buffers()):
                    if ba.is_floating_point():
                        ba.add_(bb).mul_(0.5)
        _remove_region(gm, entry_b, add_b, members_b)

    def can_be_infulenced(self, by_action):
        return False

    @staticmethod
    def generate_all_actions(
        traced: TracedModel,
        *,
        grow: bool = True,
        shrink: bool = True,
    ) -> list["ResidualBlockAction"]:
        """One duplicate per block; shrink merges identical neighbours, otherwise removes a block."""
        actions: list[ResidualBlockAction] = []
        valid: list[tuple[fx.Node, fx.Node, set[fx.Node]]] = []
        for entry, exit_node, members, reason in _discover(traced):
            if reason:
                logger.debug("ResidualBlock skip %s: %s", exit_node.name, reason)
                continue
            valid.append((entry, exit_node, members))
            if grow:
                actions.append(ResidualBlockAction(["duplicate", entry.name, exit_node.name]))
        if not shrink:
            return actions
        paired: set[fx.Node] = set()
        for (entry_a, exit_a, _), (entry_b, exit_b, _) in _adjacent_pairs(traced.gm, valid):
            paired.update((exit_a, exit_b))
            actions.append(ResidualBlockAction(["merge", entry_a.name, exit_a.name, entry_b.name, exit_b.name]))
        if len(valid) > 1:  # never remove the last residual block
            for entry, exit_node, _ in valid:
                if exit_node not in paired:
                    actions.append(ResidualBlockAction(["remove", entry.name, exit_node.name]))
        return actions

    def __str__(self):
        return f"Residual Block Action: {self.params}"
