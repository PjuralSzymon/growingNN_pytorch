File: `growingnn/utils/fx/graph_editor.py`. Class `ModelStructureEditor`. Edits `fx.GraphModule` in place, then `lint` and `recompile`. 

Called from grow/shrink actions in `growingnn/actions/` (`add_res_layer.py`, `add_res_conv_layer.py`, `add_seq_layer.py`, `add_seq_conv_layer.py`, `delete_layer.py`) via [[Torch.fx]] Layer names needs to be unique, during adding new layers protection over uniquness is manged by unique_call_module_name in [[Node analysis]].

---

## Helpers

- `_insert_call_module_after(gm, insert_after, module_name, module_input)` — new `call_module` node
- `_path_dst_to_src(dst, src)` — backward DFS along `all_input_nodes` for sequential insert
- `ModuleResolver.find_call_module` — resolve target string to `fx.Node`

---

## `add_new_residual_layer(gm, src_name, dst_name, new_layer, name)`

1. `gm.add_module(name, new_layer)`
2. `new_out = call_module(name, args=(src,))` inserted after `dst`
3. `added = operator.add(dst, new_out)`
4. Replace all uses of `dst` with `added`; set `added.args = (dst, new_out)`

Forward meaning: output of `dst` becomes `dst + proj(src)`. Shapes must broadcast; callers filter bad conv pairs at generation time using [[Graph analysis]] shape helpers.

---

## `add_new_seq_layer(gm, src_name, dst_name, new_layer, name)`

1. Find path from `dst` back to `src`
2. Insert new module after `path[1]` (node just before `dst` on that path)
3. `NodeEditor.swap_node_input(dst, src, new_out)` so `dst` reads from the new module

Handles activations and other ops between two editable endpoints. Sequential insert logic is documented on the vault action pages that call this method (see [[Torch.fx]]).

---

## `delete_layer(gm, layer_id)`

1. Merge multiple inputs with nested `operator.add`
2. Rewire each user to the merged tensor
3. `graph.erase_node(layer_node)`
4. `delattr(gm, layer_id)` when `hasattr(gm, layer_id)` — fails for dotted ids like `layer1.0.conv1`
