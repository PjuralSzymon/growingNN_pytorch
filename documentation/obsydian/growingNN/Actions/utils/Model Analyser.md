This file is about `growingnn/actions/utils/model_analyser.py`. It reads a model or an `fx.GraphModule` and answers graph questions. It does not edit the graph. Actions use it to know where to add or delete layers.

It is used by [[Residual Linear Actions]], [[Residual Conv Action]], [[Sequential Linear Actions]], [[Sequential Conv Action]], and [[Del Layer Action]]. Shape checks for residual conv pairs use [[FX Shape Probe]] from `AddResConvLayer.generate_all_actions` in `growingnn/actions/add_res_conv_layer.py` (lines 37 to 39, 50 to 54). Submodule lookup follows [[Dotted Module Names in torch.fx]] via `get_layer_module` and `nn.Module.get_submodule`. Graph edits use [[Model Transformer]]. New layer objects use [[Layer Factory]] and [[Name factory]].

---

### `get_layer_module(target, gm)`

What it does. It turns an FX `call_module` target string (or an `fx.Node`) into the real `nn.Module` on `gm`. Why. Dotted names like `layer1.0.conv1` need path walking. One `getattr(gm, "layer1.0.conv1")` call fails because the dot is not a path separator in Python attribute lookup. Where. Called from `AddResLayer.generate_all_actions`, `AddResConvLayer.generate_all_actions`, `AddSeqLayer.generate_all_actions`, `AddSeqConvLayer.generate_all_actions`, and from `has_same_output_shape` / `has_same_input_shape` / `get_common_output_shape` / `get_common_input_shape` in `growingnn/actions/delete_layer.py`.

Technicalities. Signature: `target` may be `fx.Node` or `str`. Implementation uses `gm.get_submodule(str(name))` inside `try` / `except AttributeError` and returns `None` if missing. `_is_editable_module` calls `get_layer_module(node, gm)` then checks `isinstance(module, t)` for each `t` in `EDITABLE_MODULES` from `growingnn/core/config.py`.

---

### `module_dependency_pairs(model)`

What it does. It lists every pair `(ancestor_id, descendant_id)` of editable modules where the descendant is reachable forward in the FX graph from the ancestor, and at least one of the two endpoints counts as hidden in the sense of `_is_hidden_module`. Why. Residual actions need many candidate skips, not only neighbours. Where. `AddResLayer.generate_all_actions` and `AddResConvLayer.generate_all_actions` in `growingnn/actions/add_res_layer.py` (line 30) and `add_res_conv_layer.py` (line 38).

Technicalities. The graph is taken with `gm = model` if already `fx.GraphModule`, else `torch.fx.symbolic_trace(model)` at about line 91 in `model_analyser.py`. The walk uses each node’s `.users`. Pairs are deduplicated with `dict.fromkeys`. Example for a line `l1 -> l2 -> l3` with the right hidden flags: you get `(l1,l2)`, `(l1,l3)`, `(l2,l3)`. Reachability is not the same as equal tensor shape; see [[Residual Conv Action]] and [[FX Shape Probe]] for conv residual filtering.

---

### `module_sequential_pairs(model)`

What it does. It lists `(a,b)` when `b` is the first editable `call_module` found forward from `a` along user edges, with the same hidden rule as dependency mode. Why. Sequential inserts need the next layer in order, not all transitive pairs. Where. `AddSeqLayer.generate_all_actions` and `AddSeqConvLayer.generate_all_actions`.

Technicalities. Same `gm` construction as `module_dependency_pairs`. Example chain `l1 -> l2 -> l3` yields `(l1,l2)` and `(l2,l3)` only.

---

### `get_all_hidden_modules(model)`

What it does. It returns string ids of `call_module` nodes that pass `_is_hidden_module`. Why. Delete actions enumerate hidden linear blocks that may be removed. Where. `DelLayer.generate_all_actions` in `growingnn/actions/delete_layer.py` (line 42).

---

### `_is_hidden_module(node)`

What it does. It returns `True` when the node has users, has inputs, is not wired straight from placeholder to output, and has at least one input path that already passed through another module. Why. That matches the idea of a middle layer, not pure input or pure output. Where. Used inside `module_dependency_pairs`, `module_sequential_pairs`, and `get_all_hidden_modules`. The delete doc [[Del Layer Action]] points here. ^7a8eff

---

### `get_amount_of_parameters(model)`

What it does. It returns `sum(p.numel() for p in gm.parameters())` after tracing if needed. Why. Callers track parameter count during architecture search.

---

### `get_input_layers` / `get_output_layers`

What they do. They read the undirected adjacency built from `module_sequential_pairs` and return predecessor or successor ids for one `layer_id`. Why. Delete checks need immediate linear neighbours. Where. `DelLayer.generate_all_actions`.
