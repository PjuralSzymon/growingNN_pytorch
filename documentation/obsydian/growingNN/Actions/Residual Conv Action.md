
This page is about `growingnn/actions/add_res_conv_layer.py` and the class `AddResConvLayer`. It adds a residual branch with a convolutional projection between two FX `call_module` targets, then recompiles the graph.

Hub: [[Index]].

It depends on [[Model Analyser]] for `module_dependency_pairs` and `get_layer_module`. It depends on [[FX Shape Probe]] for `call_module_output_shapes`. Execution calls `add_new_residual_layer` in [[Model Transformer]]. Factories live in [[Layer Factory]] (`ConvFactory.create_zero_conv`, `ConvFactory.create_zero_conv_before_linear`). Names come from [[Name factory]]. Width divisibility uses [[Conv to linear adapter]].

---

## Generating actions

`generate_all_actions` builds one `GraphModule` reference `gm` with `model if isinstance(model, fx.GraphModule) else torch.fx.symbolic_trace(model)` so pairs and shape keys match the same graph (see lines 37 to 38 in `add_res_conv_layer.py`).

It reads pairs from `module_dependency_pairs(gm)`. It reads shapes from `call_module_output_shapes(gm)`. For each `(layer_from_id, layer_to_id)` it loads modules with `get_layer_module(layer_from_id, model)` and `get_layer_module(layer_to_id, model)`.

Class constants: `SUPPORTED_MODULES_FROM_LAYER = (nn.modules.conv._ConvNd,)`. `SUPPORTED_MODULES_TO_LAYER = (nn.modules.conv._ConvNd, nn.modules.Linear)`.

If `layer_to` is conv and `out_shapes` is non-empty, it skips the pair unless `out_shapes[layer_from_id] == out_shapes[layer_to_id]` (lines 50 to 54). That removes cross-stage ResNet pairs that would break `torch.add`.

If `layer_to` is conv, it appends `AddResConvLayer([...])` with `ConvFactory.create_zero_conv(..., stride=1, padding=layer_from.padding, kernel_size=layer_from.kernel_size, in_channels=layer_from.out_channels, out_channels=layer_to.out_channels)`.

If `layer_to` is linear, it checks `can_insert_conv_before_linear` from `growingnn/actions/utils/conv_to_linear_adapter.py`, then may append with `ConvFactory.create_zero_conv_before_linear`.

---

## Executing actions

`execute` calls `add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])` with string ids for source and destination modules, the new conv module, and the new submodule name.

---

## Comparison with the original growingNN paper

The paper (same DOI as in [[Residual Linear Actions]], 10.1007/978-3-031-63749-0_25) treats growth as search over moves. This action is one move type. The paper does not name `ShapeProp`; this repo adds a concrete guard so random valid graphs do not pick shape-invalid conv residuals.

---

## Known limitations

1. Conv to linear path has no `call_module_output_shapes` equality check in `add_res_conv_layer.py` lines 64 to 76; execute-time errors are still possible there.

2. When `call_module_output_shapes` returns `{}`, conv-conv filtering is disabled; rare models may still throw on add.

3. `generate_all_actions` uses `torch.fx.symbolic_trace` for a plain `nn.Module` input while your project may elsewhere prefer `growingnn.core.fx_trace.trace`; submodule names should match if you always pass the same traced `gm` into both analyse and execute (as in `tests/regression/resnet_regression_test.py` line 78).
