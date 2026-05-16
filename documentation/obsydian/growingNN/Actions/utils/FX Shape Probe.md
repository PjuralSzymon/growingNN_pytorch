This note is about `growingnn/actions/utils/fx_shape_probe.py`. The file has three functions. One public: `call_module_output_shapes`. Two helpers: `_node_output_shape_tuple` and `_default_probe_tensor`.

### Overall idea (why this module exists)

When we add a residual conv connection, we sum two activation tensors: `torch.add(dst, proj(src))` in `add_new_residual_layer` (`growingnn/actions/utils/model_transformations.py`).

[[Model Analyser]] can list pairs `(layer_from, layer_to)` that are reachable in the FX graph. Reachability only means there is a path in the graph. It does not mean the two tensors have the same shape.

Example on ResNet-18. `layer3.0.conv2` can reach `layer4.1.conv1` in the graph. After a stride-2 stage the later conv often has smaller height and width. The new `proj` conv uses stride 1, so its output keeps the source spatial size. Then `torch.add` fails with a shape mismatch `RuntimeError`.

This module runs PyTorch `ShapeProp` once. It stores the output shape of each `call_module` node. [[Residual Conv Action]] uses that map to drop conv-to-conv pairs whose shapes differ before it builds actions.

---

### `_node_output_shape_tuple(node)` (lines 10 to 18)

What it does. After `ShapeProp` runs, each FX `node` gets metadata in `node.meta`. This helper reads the output shape of one node as a plain tuple of ints, e.g. `(1, 64, 56, 56)`.

How. It looks at `node.meta["val"]` first. If that object has a `.shape` attribute, it copies the shape into a tuple. If not, it tries `node.meta["tensor_meta"].shape` (another layout `ShapeProp` may write). If neither works, it returns `None`.

Why. `call_module_output_shapes` walks all `call_module` nodes and needs one small function to pull a shape out of meta without caring which key `ShapeProp` filled.

---

### `_default_probe_tensor(gm)` (lines 21 to 30)

What it does. Builds a fake input tensor so `ShapeProp` can run a forward pass on the graph.

How.

1. It checks that `gm` has at least one `placeholder` node (the graph input). If there is no placeholder, it returns `None`.
2. It picks `device` and `dtype` from the first parameter of `gm`, or CPU float32 if the model has no parameters.
3. It returns `torch.randn(1, 3, 224, 224, ...)` — batch 1, 3 channels, spatial size 224×224.

Why. `ShapeProp` needs a real tensor to propagate shapes. Callers of `call_module_output_shapes` often pass only `gm`. This default matches typical ImageNet-style conv nets. It is not guaranteed to match every model input (see Known limitations).

---

### `call_module_output_shapes(gm, example=None)` (lines 33 to 59)

What it does. Runs shape propagation and returns a dictionary:

`{ "layer_name": (dim0, dim1, ...) , ... }`

Keys are `call_module` target strings (same ids as in the FX graph, e.g. `layer3.0.conv2`). Values are output activation shapes for that module on the probe input.

How.

1. `probe = example` if you passed `example`, else `_default_probe_tensor(gm)`.
2. If `probe` is `None`, return `{}`.
3. Call `ShapeProp(gm).propagate(probe)`. On any exception, return `{}` (fail open: no shape filter).
4. Loop `gm.graph.nodes`. For each `call_module` with a string `target`, call `_node_output_shape_tuple(node)` and store the shape in the dict.

Why. [[Residual Conv Action]] needs a cheap static check before proposing residual conv actions. Equal shapes for conv-to-conv pairs mean `torch.add` is plausible at the default probe resolution.

---

### Where it is used

`AddResConvLayer.generate_all_actions` in `growingnn/actions/add_res_conv_layer.py` (lines 37 to 39, 49 to 54).

1. `pairs = module_dependency_pairs(gm)` from [[Model Analyser]].
2. `out_shapes = call_module_output_shapes(gm)`.
3. For each pair where `layer_to` is a conv (`nn.modules.conv._ConvNd`), if `out_shapes` is non-empty, it keeps the pair only when `out_shapes[layer_from_id] == out_shapes[layer_to_id]`. Missing keys skip the pair too.

Conv-to-linear branches (lines 64 onward) do not use this equality check. They use [[Conv to linear adapter]] for channel divisibility only.

Also linked from [[Model Analyser]] and [[Residual Conv Action]]. Unit code: `tests/unit/actions/utils/fx_shape_probe_test.py` with `ModelFactory.simple_conv_chain_2` in `tests/model_factory.py`.

---

### Comparison with the original growingNN paper

Chapter DOI `10.1007/978-3-031-63749-0_25` does not name FX `ShapeProp`. The idea matches the paper: only propose architecture moves that keep tensor math valid. Here the check is explicit and only wired into conv residual action generation.

---

### Known limitations

1. Default probe is fixed at `(1, 3, 224, 224)`. Models with other input ranks or channel counts may make `ShapeProp` fail. Then the function returns `{}` and conv-conv filtering is turned off in `add_res_conv_layer.py`.

2. Shapes depend on input size. A pair that matches at 224×224 may not match at 64×64. Regression `tests/regression/resnet_regression_test.py` runs forwards with `INPUT_SHAPE = (3, 64, 64)` but action listing still probes with 224×224 unless you change line 30 in `fx_shape_probe.py`.

3. Conv-to-linear residuals are not filtered by this module. Bad pairs there can still fail at execute time.

---

### Related

[[Model Analyser]], [[Residual Conv Action]], [[Model Transformer]], [[Conv to linear adapter]], [[Dotted Module Names in torch.fx]].
