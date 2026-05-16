Code: `growingnn/actions/add_res_conv_layer.py` (`AddResConvLayer`). Residual conv branch between two FX modules.

Uses [[Model Analyser]] `module_dependency_pairs`, `get_layer_module`. Uses [[Layer Analyser]] for all width rules. Executes `add_new_residual_layer` in [[Model Transformer]]. Factories: [[Layer Factory]] `ConvFactory.create_zero_conv`, `create_zero_conv_before_linear`. Names: [[Name factory]].

---

## Generating actions

One `gm`, then `out_shapes` and `in_shapes` from `LayerShapeAnalyser` (lines 24 to 26).

For each dependency pair:

1. Conv → conv. `find_equal_conv_output_shapes(s_from, s_to)` must be true (equal 4D tuples). Then `create_zero_conv` with `layer_from.out_channels`, `layer_to.out_channels`, same `kernel_size` and `padding`, `stride=1`.

2. Conv → linear. `find_conv_before_linear_sizes(s_from, in_shapes[to], s_to, for_residual=True)` returns `(channels, linear_out)`. Then `create_zero_conv_before_linear` (pool + flatten inside the new module).

No call to `can_insert_conv_before_linear` in this file; divisibility is inside [[Layer Analyser]] `find_conv_before_linear_sizes`.

---

## Executing actions

`execute` → `add_new_residual_layer` with conv or `Sequential` submodule as the new branch.

---

## Comparison with the original growingNN paper

Search moves in DOI 10.1007/978-3-031-63749-0_25 include adding parallel paths. This action is the conv variant. `ShapeProp` filtering is extra in this PyTorch port to avoid `layer3` → `layer4` style size mismatches on ResNet-18.

Conv → linear residual still uses pool+flatten inside the new module (unlike [[Sequentail Linear Actions]] sequential insert).

---

## Known limitations

1. If shape maps are empty, conv→conv filter is off and runtime `torch.add` may still fail.
2. `get_layer_module` still needed for channel and kernel metadata.
3. Sequential conv→linear growth is not here; see [[Sequentail Linear Actions]].

---

## Related

[[Layer Analyser]], [[Conv to linear adapter]], [[Sequential Conv Action]], [[Sequentail Linear Actions]].
