[[Torch.fx]]

File: `growingnn/utils/fx/sum_nodes.py`.

Variadic residual sums in FX graphs. Residual grow calls `connect_residual_branch` from `ModelStructureEditor.add_new_residual_layer` in [[Graph editor]]. Layer delete calls `remove_layer_from_sums` from `ModelStructureEditor.delete_layer`.

---

## `nary_add`

`nary_add(*tensors)` sums tensors left to right. Registered as `call_function` target in the FX graph. Replaces nested `operator.add` chains with one node per sum.

---

## `connect_residual_branch(gm, dst, src, module_name)`

Used when adding a residual skip.

1. Insert `call_module(module_name, args=(src,))` before or after the existing sum on `dst`.
2. Flatten existing sum terms with `_flatten_terms`.
3. Append the new branch tensor.
4. `_install_sum` creates a new `nary_add` and replaces the old sum or `dst` output.

`_install_sum` inserts the new sum after `_latest_node(gm, terms)` so every term is defined before use in topological order.

---

## `is_merge_branch_layer(node)`

Returns `True` when the node has users and every user is a sum node (`is_sum_node`).

Such layers are residual side branches only. Layer delete removes them by dropping the branch from sums, not by bypass wiring into the trunk.

---

## `remove_layer_from_sums(gm, layer_node)`

Called from `ModelStructureEditor.delete_layer` for merge-branch layers.

For each sum user of `layer_node`:

1. Flatten sum terms.
2. Drop `layer_node` from the term list.
3. If one term remains, `replace_all_uses_with` that term and erase the sum.
4. Else rebuild the sum with `_install_sum` and `_erase_dead_sums`.

---

## Comparison with the original growingNN paper

The paper describes residual connections as graph edits. R5 stores them as explicit `nary_add` nodes so FX `lint` can check use-before-def and `GraphConnectivity` in [[Graph analysis]] can walk one output path.

---

## Known limitations

1. Only `operator.add` and `nary_add` count as sum nodes. Custom add wrappers are ignored.

2. Removing the last term from a sum raises `ValueError`; delete generation should not emit such actions.

3. Sum rebuild does not itself prune orphaned upstream branches; `prune_unreachable_nodes` in [[Graph editor]] runs after `delete_layer`.
