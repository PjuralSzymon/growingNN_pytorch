File: `growingnn/actions/utils/model_analyser.py`. Reads an `nn.Module` or `fx.GraphModule`. Does not edit the graph.

Used by all growth actions and [[Del Layer Action]]. Shape guards use [[Layer Analyser]]. Tests: `tests/unit/actions/utils/model_analyser_test.py`.

---

### `get_layer_module(target, gm)`

Resolves a `call_module` target or `fx.Node` via `gm.get_submodule(str(name))` (works for flat names like `l1` and qualified paths like `layer1.0.conv1`). Returns `None` on `AttributeError`.

Where: `AddResConvLayer`, `AddSeqConvLayer` (kernel_size, padding, channels). Not used by [[Del Layer Action]] for width checks anymore.

---

### `module_dependency_pairs(model)`

All `(ancestor, descendant)` editable pairs where the descendant is reachable forward and at least one endpoint is hidden. Used by [[Residual Linear Actions]] and [[Residual Conv Action]].

Deduped with `dict.fromkeys`. Logs only `number of dependency pairs` at DEBUG (per-pair spam removed).

---

### `module_sequential_pairs(model)`

First editable module forward from each source along user edges (same hidden rule). Used by [[Sequentail Linear Actions]] and [[Sequential Conv Action]].

Example `l1 → l2 → l3` yields `(l1,l2)`, `(l2,l3)` only.

---

### `get_all_hidden_modules(model)`

Lists `call_module` targets that pass `_is_hidden_module`. Used by [[Del Layer Action]].

---

### `_is_hidden_module(node)`

Middle layer: has users and inputs, not placeholder-to-output only.

---

### `get_input_layers` / `get_output_layers`

Built from one `module_sequential_pairs` pass inside `_sequential_adj`. Immediate sequential preds or succs for one `layer_id`.

---

### `get_amount_of_parameters(model)`

`sum(p.numel() for p in gm.parameters())`. Used in `tests/regression/resnet_regression_test.py` and regression plots.

---

### Related

[[Layer Analyser]], [[Model Transformer]], [[Layer Factory]], [[Name factory]].
