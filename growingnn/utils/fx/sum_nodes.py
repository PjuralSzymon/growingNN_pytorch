"""Variadic nary_add sums in FX graphs."""

from __future__ import annotations

import operator

import torch
import torch.fx as fx


def nary_add(*tensors: torch.Tensor) -> torch.Tensor:
    if not tensors:
        raise ValueError("nary_add requires at least one tensor")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result + tensor
    return result


_SUM_TARGETS = (operator.add, nary_add)


def is_sum_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target in _SUM_TARGETS


def _flatten_terms(node: fx.Node) -> list[fx.Node]:
    if not is_sum_node(node):
        return [node]
    terms: list[fx.Node] = []
    for arg in node.args:
        terms.extend(_flatten_terms(arg) if is_sum_node(arg) else [arg])
    return terms


def _find_sum_user(node: fx.Node) -> fx.Node | None:
    return next((user for user in node.users if is_sum_node(user)), None)


def _create_sum(gm: fx.GraphModule, terms: list[fx.Node], *, after: fx.Node) -> fx.Node:
    with gm.graph.inserting_after(after):
        return gm.graph.call_function(nary_add, args=tuple(terms))


def _erase_dead_sums(gm: fx.GraphModule, node: fx.Node) -> None:
    if not is_sum_node(node):
        return
    children = [arg for arg in node.args if is_sum_node(arg)]
    gm.graph.erase_node(node)
    for child in children:
        if len(child.users) == 0:
            _erase_dead_sums(gm, child)


def _install_sum(
    gm: fx.GraphModule,
    terms: list[fx.Node],
    *,
    after: fx.Node,
    replace: fx.Node,
) -> fx.Node:
    """Create nary_add(*terms) and rewire *replace* users; args are fixed after rewire."""
    new_sum = _create_sum(gm, terms, after=after)
    replace.replace_all_uses_with(new_sum)
    new_sum.args = tuple(terms)
    return new_sum


def connect_residual_branch(
    gm: fx.GraphModule,
    dst: fx.Node,
    src: fx.Node,
    module_name: str,
) -> None:
    """Insert *module_name* from *src* and sum it with *dst* in one nary_add."""
    existing = _find_sum_user(dst)
    insert_ctx = (
        gm.graph.inserting_before(existing)
        if existing is not None
        else gm.graph.inserting_after(dst)
    )
    with insert_ctx:
        branch = gm.graph.call_module(module_name, args=(src,))

    terms = _flatten_terms(existing) if existing else [dst]
    terms.append(branch)
    if existing is not None:
        _install_sum(gm, terms, after=branch, replace=existing)
        _erase_dead_sums(gm, existing)
    else:
        _install_sum(gm, terms, after=branch, replace=dst)


def sum_nodes(gm: fx.GraphModule, terms: list[fx.Node]) -> fx.Node:
    """Create one nary_add node for *terms*."""
    return _create_sum(gm, terms, after=terms[-1])
