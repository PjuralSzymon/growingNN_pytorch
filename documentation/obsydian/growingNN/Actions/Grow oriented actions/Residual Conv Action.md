[[Actions]]

This page is about `growingnn/actions/add_res_conv_layer.py` and the class `AddResConvLayer`. It adds a residual branch with a convolutional projection between two FX `call_module` targets, then recompiles the graph.

It depends on [[Torch.fx]]: `GraphStructureQuery.module_dependency_pairs`, `ModuleResolver.get_layer_module`, `LayerShapeAnalyser.get_layer_output_shapes`, `LayerBridgeFinder`, `ModelStructureEditor.add_new_residual_layer`, `ModuleResolver.unique_call_module_name`. Factories in [[Layer Factory]] (`ConvFactory.create_zero_conv`, `ConvFactory.create_zero_conv_before_linear`). Width divisibility uses [[Conv to linear adapter]].

---

## Exclusion cases

For each dependency pair from `module_dependency_pairs(gm)`:

1. if `find_equal_conv_output_shapes` is false and `find_conv_before_linear_sizes(..., for_residual=True)` returns `None` then skip (neither identical 4-D conv outputs for a zero conv skip, nor a conv→linear bridge where `linear_in % conv_channels == 0` and linear output width is known)
2. for the conv→conv path only: `find_equal_conv_output_shapes` is false when probed 4-D shapes differ or are missing (residual add would not broadcast)
3. for the conv→linear path only: `find_conv_before_linear_sizes` returns `None` when conv channels or linear dims are missing, or `linear_in % channels != 0` (flattened conv features cannot align to linear input)

---

## Generating actions

`generate_all_actions` probes output and input shapes once, then walks dependency pairs.

Equal-shape conv→conv pairs get `ConvFactory.create_zero_conv`. Other conv→linear pairs get `ConvFactory.create_zero_conv_before_linear` when `find_conv_before_linear_sizes` returns sizes.

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

3. `generate_all_actions` uses `torch.fx.symbolic_trace` for a plain `nn.Module` input while your project may elsewhere prefer `growingnn.core.fx_trace.trace`; submodule names should match if you always pass the same traced `gm` into both analyse and execute.
