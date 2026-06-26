This page is about `growingnn/actions/add_res_linear_layer.py` and the class `AddResLinearLayer`.

It uses [[Torch.fx]]: `GraphStructureQuery.module_dependency_pairs`, `ModuleResolver.get_layer_module`, `ModelStructureEditor.add_new_residual_layer`, `ModuleResolver.unique_call_module_name`. New layers from [[Layer Factory]]. Related conv variant: [[Residual Conv Action]].

---

## Generating actions

`AddResLinearLayer.generate_all_actions(model, layer_types=...)` reads pairs from `module_dependency_pairs(model)` from  [[Torch.fx]]  at line 30 in `add_res_linear_layer.py`. For each `(layer_from_id, layer_to_id)` it resolves modules with `get_layer_module(layer_from_id, model)` and `get_layer_module(layer_to_id, model)`.

It keeps only pairs where both ends pass `isinstance(..., AddResLinearLayer.SUPPORTED_MODULES)`. Today `SUPPORTED_MODULES = (nn.Linear,)` at line 15. The second argument to `isinstance` must be a tuple, not a list, so Python accepts it (see `TypeError` fix in recent work on `isinstance` arg 2).

For each kept pair and each `Layer_Type` in `layer_types`, it builds a linear projector with `layer_from.out_features` and `layer_to.out_features`, picks a name via `unique_call_module_name`, and appends `AddResLinearLayer([layer_from_id, layer_to_id, layer, name])`.

---

## Executing actions

`execute` forwards to `add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])` at line 18 in `add_res_linear_layer.py`.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 describes dynamic architecture search with MCTS-style ideas. Linear residual growth here is one concrete operator in that family. Initialization differs: the old note below still applies about `LinearFactory` versus a single global init mode in some older code paths.

The original paper doesn't focus on conv layer initialization; it is using global config mode, which can be uniform or normal draws, which was one possible reason for data loss in the paper plots. This repo uses targeted init in `LinearFactory` to limit shock when a new skip appears.

---

## Known limitations

1. Only `nn.Linear` subclasses pass `SUPPORTED_MODULES`; lazy or quantized linears need review if you add them to the tuple.

### Plot from regression

Graph how output of model changed compared with how many parameters the model has during adding linear residual layers.
![[Pasted image 20260510215912.png]]
