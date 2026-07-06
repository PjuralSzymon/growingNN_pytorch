[[Torch.fx]]

File: `growingnn/utils/fx/graph_editor.py`. Class `ModelStructureEditor`. Edits `fx.GraphModule` in place, then `lint` and `recompile`.



Called from grow/shrink actions in `growingnn/actions/`. Layer names must be unique; `ModuleResolver.unique_call_module_name` in [[Node analysis]] handles collisions.



Residual wiring uses `connect_residual_branch` from `sum_nodes.py`. Layer delete uses `remove_layer_from_sums`, `compute_bypass_matching`, and `prune_unreachable_nodes`.



---



## Helpers



- `_insert_call_module_after(gm, insert_after, module_name, module_input)` — new `call_module` node

- `_path_dst_to_src(dst, src)` — backward DFS along `all_input_nodes` for sequential insert

- `bypass_shapes_compatible(pred_shape, succ_shape)` — equal activation tuples

- `compute_bypass_matching(input_layers, output_layers, output_shapes, input_shapes)` — map each successor id to one compatible predecessor id

- `branch_only_bypass_compatible(layer_node, input_shapes)` — true when a layer with no sequential successor can be skipped via one FX input

- `_producer_before_layer`, `_rewire_layer_users`, `_rewire_branch_only_layer` — bypass rewire helpers

- `prune_unreachable_nodes(gm)` — erase nodes not on a path to `output` (see `GraphConnectivity` in [[Graph analysis]])



---



## `add_new_residual_layer(gm, src_name, dst_name, new_layer, name)`



1. `gm.add_module(name, new_layer)`

2. `connect_residual_branch(gm, dst_node, src_node, name)` from `sum_nodes.py`

3. `lint` and `recompile`



Forward meaning: output at `dst` becomes `nary_add(dst, new_layer(src))`.



---



## `add_new_seq_layer(gm, src_name, dst_name, new_layer, name)`



1. Find path from `dst` back to `src`

2. Insert new module after `path[1]` (node just before `dst` on that path)

3. `NodeEditor.swap_node_input(dst, src, new_out)` so `dst` reads from the new module



---



## `delete_layer(gm, layer_id)`



Used by `DelLayer` in `delete_layer.py`.



1. Resolve `layer_node`, `input_layers`, `output_layers`, and shape maps from [[Graph analysis]].

2. If `is_merge_branch_layer(layer_node)` (`sum_nodes.py`): `remove_layer_from_sums(gm, layer_node)`.

3. Else if no sequential successors: `_rewire_branch_only_layer` when `branch_only_bypass_compatible` passes.

4. Else: `_rewire_layer_users` with `compute_bypass_matching`.

5. `graph.erase_node(layer_node)` and drop submodule when top-level.

6. `prune_unreachable_nodes(gm)` — removes dangling `act` nodes and dead branches not reaching the graph output.

7. `lint` and `recompile`.



Old behaviour (sum all inputs into one tensor for every user) is replaced by pairwise bypass and merge-branch removal.



---



## Known limitations



1. `delattr(gm, layer_id)` fails for dotted submodule paths; only top-level names are removed today.



2. Prune keeps exactly the subgraph backward-reachable from `output`. It does not merge parallel live paths into one trunk.



3. `branch_only_bypass_compatible` rejects layers whose users include sum nodes; those must use the merge-branch path.

