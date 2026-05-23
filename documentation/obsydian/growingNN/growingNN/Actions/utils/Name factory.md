This page is about `growingnn/actions/utils/name_factory.py` and the function `unique_call_module_name`.

### What it does

It picks a string name for a new submodule on an `nn.Module` or `fx.GraphModule` so `gm.add_module(name, layer)` does not collide with existing names or existing `call_module` targets in the FX graph.

### Algorithm

Lines 22 to 26 collect `model._modules.keys()`. If `model` is `fx.GraphModule`, it also unions the set of all `str(n.target)` for nodes with `n.op == "call_module"` (lines 23 to 26).

It scans names equal to `base` or starting with `base + "_"` plus digits (lines 28 to 35). It returns `base + "_0"` if no hit (lines 37 to 38). Otherwise it returns `base + "_" + str(max(suffixes) + 1)` (line 39).

### Where it is used

`AddResLayer.generate_all_actions` in `add_res_layer.py` line 45. `AddResConvLayer` line 48 in `add_res_conv_layer.py`. `AddSeqLayer` and `AddSeqConvLayer` use the same helper for new layer names.

### Known limitations

The scan is string based. If a name exists only inside a nested submodule but not as a top-level attribute on `gm`, collision rules depend on what `_modules` and the FX graph list expose for your trace style.

### Related

[[Model Transformer]], [[Residual Linear Actions]], [[Residual Conv Action]].
