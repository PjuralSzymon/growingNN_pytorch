Naming for new FX submodules lives in [[Torch.fx]]: `ModuleResolver.unique_call_module_name` in `growingnn/utils/fx/node_analysis.py` (lines 38 to 55).

### What it does

Picks a string for `gm.add_module(name, layer)` that does not collide with `model._modules` or existing `call_module` targets on the graph.

### Algorithm

Collects `model._modules.keys()` and, for `GraphModule`, all `str(n.target)` for `call_module` nodes. Scans `base`, `base_0`, `base_1`, … and returns the next free suffix.

### Where it is used

`AddResLinearLayer`, `AddResConvLayer`, `AddSeqLinearLayer`, `AddSeqConvLayer` during `generate_all_actions`.