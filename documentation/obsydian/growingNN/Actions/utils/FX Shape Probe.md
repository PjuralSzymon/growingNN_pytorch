This note is about `growingnn/actions/utils/fx_shape_probe.py`. It runs `torch.fx.passes.shape_prop.ShapeProp` once on an `fx.GraphModule` and records output shapes for each `call_module` target string.

Why it exists. `module_dependency_pairs` in `growingnn/actions/utils/model_analyser.py` only knows graph reachability. A pair like `layer3.0.conv2` to `layer4.1.conv1` on ResNet-18 can be reachable while the two tensors have different height and width. `add_new_residual_layer` in `growingnn/actions/utils/model_transformations.py` builds `torch.add(dst, proj(src))`. For conv to conv, `proj` uses stride 1, so its spatial size matches the source conv output. The destination conv output can be smaller after a stride-2 stage. Then `torch.add` raises `RuntimeError` on shape mismatch. The probe removes those pairs before action generation.

Where it is used. `AddResConvLayer.generate_all_actions` in `growingnn/actions/add_res_conv_layer.py` calls `call_module_output_shapes(gm)` after `module_dependency_pairs(gm)`. For `isinstance(layer_to, nn.modules.conv._ConvNd)`, it requires `out_shapes[from] == out_shapes[to]` when `out_shapes` is non-empty. Conv to linear branch is not filtered by this equality in the same way (line 64 onward in `add_res_conv_layer.py`).

It is linked from [[Model Analyser]], [[Residual Conv Action]], and [[ResNet18 regression script]]. Tests live in `tests/unit/actions/utils/fx_shape_probe_test.py` using `ModelFactory.simple_conv_chain_2` in `tests/model_factory.py` and `torch.fx.symbolic_trace`.

---

### Main API

`call_module_output_shapes(gm, example=None) -> dict[str, tuple[int, ...]]`

Technicalities. If `example` is `None`, `_default_probe_tensor` builds `torch.randn(1, 3, 224, 224)` on the device and dtype of the first parameter of `gm`, else CPU float32. If there is no placeholder or propagation throws, the function returns `{}`. In that case `AddResConvLayer` does not apply the conv-conv shape filter (empty dict means no extra skip). Shape is read from `node.meta["val"].shape` if present, else `node.meta["tensor_meta"].shape`, via `_node_output_shape_tuple` at lines 10 to 18 in `fx_shape_probe.py`.

---

### Comparison with the original growingNN paper

The Springer chapter 10.1007/978-3-031-63749-0_25 does not spell out FX `ShapeProp` guards. The idea matches the paper goal: only apply architecture moves that keep tensor math valid. Here the check is explicit and local to conv residual generation.

---

### Known limitations

1. Default probe `(1, 3, 224, 224)` may fail for models that need other channel counts or ranks; then `out_shapes` is `{}` and conv-conv filtering is off. Pass a custom `example` only if you extend the API; today `generate_all_actions` uses the default path only.

2. Shapes depend on input resolution. A pair valid at `224×224` might be invalid at `64×64` or the reverse. Regression `tests/regression/resnet_regression_test.py` uses `INPUT_SHAPE = (3, 64, 64)` and `BATCH_SIZE = 2` for the forward pass; the probe for action listing still uses `1×3×224×224` unless you change `fx_shape_probe.py` line 30.

3. Conv to linear residuals are not guarded by this file; bad pairs there could still fail at execute time.
