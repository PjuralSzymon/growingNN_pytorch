This note is about `growingnn/actions/utils/layer_analyser.py`. Public API: `get_layer_output_shapes`, `get_layer_input_shapes`. Not the same file as [[Model Analyser]] (graph reachability).

### Overall idea

Residual conv sums two activations: `torch.add(dst, proj(src))` in `add_new_residual_layer` ([[Model Transformer]]).

[[Model Analyser]] lists reachable layer pairs. Reachability does not imply equal tensor shape. On ResNet-18, `layer3.0.conv2` to `layer4.1.conv1` can be reachable but different H×W after stride-2. Then `torch.add` fails.

This module runs `ShapeProp` once and builds shape maps keyed by `call_module` target strings. [[Residual Conv Action]] compares output shapes for conv-to-conv pairs.

---

### `get_layer_output_shapes(gm, example=None)`

What. Returns `{layer_id: output_shape_tuple}` for every `call_module` node, e.g. `{"c1": (1, 4, 16, 16)}`.

How. Uses `example` if given, else `_default_example_input` (`randn(1, 3, 224, 224)` on model device). Runs `ShapeProp(gm).propagate`. On failure returns `{}`.

---

### `get_layer_input_shapes(gm, example=None)`

What. Returns `{layer_id: input_shape_tuple}` where the input is the shape of the first FX arg to that `call_module` (usually the previous layer's output).

How. Same propagation as output shapes. Input shape for layer L is read from `_node_shape` on `L.args[0]` when that arg is an `fx.Node`.

---

### Helpers (private)

`_node_shape`, `_default_example_input`, `_input_shape_for_layer`, `_collect_layer_shapes` — shared propagation for both public functions.

---

### Where it is used (production)

`AddResConvLayer.generate_all_actions` in `growingnn/actions/add_res_conv_layer.py` uses `get_layer_output_shapes` for conv-to-conv equality (lines 39 to 54).

`AddSeqLayer.generate_all_actions` in `growingnn/actions/add_seq_layer.py` uses `find_bridge_linear_sizes` on rank-2 activations only: `(batch, features)` → `in_features` / `out_features` are the second dim. Shapes like `(1, 512, 7, 7)` return `None` (conv path, not a flattened linear).

---

### What the conv residual check actually does (lines 50 to 54)

This is the only reason the module exists in production. The code is easy to read backwards, so here is the logic in plain form.

Setup. `out_shapes = get_layer_output_shapes(gm)`. For each conv-to-conv pair from [[Model Analyser]]:

```
s_from = out_shapes.get(layer_from_id)
s_to   = out_shapes.get(layer_to_id)
```

Step 1 — is shape filtering on? `if out_shapes:` means filtering runs only when `ShapeProp` produced a non-empty map. If propagation failed, `out_shapes` is `{}` and every conv pair is allowed (fail open).

Step 2 — skip bad pairs:

```
if s_from is None or s_to is None or s_from != s_to:
    continue
```

We do care about the full shape tuples, not only “not null”.

| Case | What happens |
|------|----------------|
| `s_from` missing | Skip — cannot compare |
| `s_to` missing | Skip — cannot compare |
| `s_from != s_to` (e.g. `(1,256,56,56)` vs `(1,512,28,28)`) | Skip — `torch.add` would fail after `proj(src)` |
| `s_from == s_to` (same tuple) | Keep — residual conv action is added |

So “not null” is necessary but not enough. Both shapes must exist and be equal element-wise as tuples. That is how we drop ResNet pairs like `layer3` → `layer4` where reachability holds but spatial size differs.

Conv-to-linear branches in the same function (lines 64 onward) do not call [[Layer Analyser]] at all. They only use [[Conv to linear adapter]] for channel divisibility.

---

### Unit tests

`tests/unit/actions/utils/layer_analyser_test.py` checks `get_layer_output_shapes` and `get_layer_input_shapes` on `ModelFactory.simple_conv_chain_2`.

---

### Comparison with the original growingNN paper

Chapter DOI `10.1007/978-3-031-63749-0_25` does not name `ShapeProp`. The check matches the paper goal: only propose moves that keep tensor math valid.

---

### Known limitations

1. Default example `(1, 3, 224, 224)` may not match every model; then maps are `{}` and conv filtering is off.
2. Shapes depend on probe resolution; regression may forward at 64×64 while probing at 224×224.
3. Conv-to-linear residuals are not filtered here.
4. Calling both getters runs `ShapeProp` twice unless you pass the same `example` and accept the cost (simple API by design).

---

### Related

[[Model Analyser]], [[Residual Conv Action]], [[Model Transformer]], [[Conv to linear adapter]], [[Dotted Module Names in torch.fx]].
