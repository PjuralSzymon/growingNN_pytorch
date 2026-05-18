This page documents `growingnn/actions/utils/model_transformations.py`. These functions edit an `fx.GraphModule` in place: they change `gm.graph`, add submodules on `gm`, then call `gm.graph.lint()` and `gm.recompile()`.

They are called from [[Residual Linear Actions]], [[Residual Conv Action]], [[Sequentail Linear Actions]], [[Sequential Conv Action]], and [[Del Layer Action]]. Names for new modules come from [[Name factory]]. New layer objects come from [[Layer Factory]]. Analysis that chooses endpoints lives in [[Model Analyser]].

---

### `_insert_call_module_after(gm, insert_after, module_name, module_input)`

Lines 7 to 9. Opens `gm.graph.inserting_after(insert_after)` and returns `gm.graph.call_module(module_name, args=(module_input,))`.

---

### `_find_call_module(nodes, target_name)`

Lines 13 to 21. Linear scan for `n.op == "call_module"` and `n.target == target_name`. Raises `ValueError` listing all `call_module` targets if missing. Used by all public edit functions before they rewrite edges.

---

### `add_new_residual_layer(gm, src_name, dst_name, new_layer, name)`

Lines 24 to 41.

Steps. `gm.add_module(name, new_layer)` at line 27. Finds `src` and `dst` nodes by string name at lines 29 to 30. Inserts `new_out = call_module(name, args=(src,))` after `dst` at line 32. Builds `added = operator.add(dst, new_out)` at lines 34 to 35. Replaces every use of `dst` with `added`, then forces `added.args = (dst, new_out)` at lines 37 to 38 so the add keeps the original `dst` tensor and the skip branch.

Meaning. Output tensor of `dst` becomes `dst + proj(src)` in forward order. Shapes of `dst` and `proj(src)` must broadcast. [[Layer Analyser]] filters some bad conv pairs at action generation time.

---

### `add_new_seq_layer(gm, src_name, dst_name, new_layer, name)`

Lines 69 to 90.

Steps. Adds submodule at line 72. Finds `src` and `dst` at lines 74 to 75. Rejects identical src and dst at lines 76 to 77. Calls `_path_dst_to_src(dst, src)` at lines 79 to 81. Takes `path[1]` as the node to insert after (line 83). Inserts `call_module(name, args=(src,))` after that node (line 85). Rewires `dst` so its input edge that used to come from `src` now comes from `new_out` via `_replace_node_input` at line 87.

Meaning. Inserts one new module on the path between two sequential endpoints even if non-module ops sit between them. See [[Sequentail Linear Actions]].

---

### `delete_layer(gm, layer_id)`

Lines 93 to 125. ^f4531d

Steps. Finds the `call_module` node for `layer_id` at lines 96 to 99. Collects `input_nodes` and `output_nodes` at lines 101 to 102. Merges multiple inputs with nested `operator.add` at lines 105 to 113. For each user, replaces inputs that pointed to the deleted node with `new_input` at lines 114 to 115. Erases the node at line 118. Removes submodule with `if hasattr(gm, layer_id): delattr(gm, layer_id)` at lines 119 to 120.

Known bug shape. For dotted `layer_id` such as `layer1.0.conv1`, `hasattr(gm, layer_id)` is false because there is no single attribute with dots in the name. Submodule removal may fail while the graph node is already erased. Prefer `gm.delete_submodule(layer_id)` in a try block for future fix. Documented here so [[Del Layer Action]] readers know the risk.

---

### Helpers `_path_dst_to_src`, `_replace_node_input`

Lines 45 to 66. DFS backward from `dst` to `src` along `all_input_nodes`. `_replace_node_input` edits `args` and `kwargs` tuples on a user node.

---

### Comparison with the original growingNN paper

The paper describes dynamic graphs and mutations during training. This file is the low-level FX edit layer: small deterministic rewrites instead of hand-written `forward` for every variant.

---

### Known limitations

1. `delete_layer` submodule removal with `hasattr` or `delattr` does not support dotted names (see above).

2. `add_new_residual_layer` does not check shapes; callers must filter (see [[Layer Analyser]] for conv).

3. All functions assume `gm` is already a traced `GraphModule` with stable `call_module` targets.

### Related

[[Model Analyser]], [[Name factory]], [[Layer Factory]], [[Del Layer Action]].
