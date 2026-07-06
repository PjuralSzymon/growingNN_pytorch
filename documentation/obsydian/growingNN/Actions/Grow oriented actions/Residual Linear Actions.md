[[Actions]]

This page is about `growingnn/actions/add_res_linear_layer.py` and the class `AddResLinearLayer`.

It uses [[Torch.fx]]: `GraphStructureQuery.module_dependency_pairs`, `ModuleResolver.get_layer_module`, `ModelStructureEditor.add_new_residual_layer`, `ModuleResolver.unique_call_module_name`. New layers from [[Layer Factory]]. Related conv variant: [[Residual Conv Action]].

---

## Exclusion cases

For each dependency pair from `module_dependency_pairs(gm)`:

1. if `find_bridge_res_linear_sizes` returns `None` then skip (cannot read matching linear in/out feature dims from probed output shapes at both ends of the skip)
2. if `in_features * out_features` exceeds `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE` then skip (residual projector weight matrix would exceed the safety cap)

---

## Generating actions

`AddResLinearLayer.generate_all_actions(model, layer_types=...)` walks dependency pairs. For each pair that passes exclusion, it emits one action per `Layer_Type` in `layer_types` with a `LinearFactory.create_linear` projector and a unique `res_linear_{TYPE}` name.

---

## Executing actions
`execute` forwards to `add_new_residual_layer(model, self.params[0], self.params[1], self.params[2], self.params[3])` at line 18 in `add_res_linear_layer.py`.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 describes dynamic architecture search with MCTS-style ideas. Linear residual growth here is one concrete operator in that family. Initialization differs: the old note below still applies about `LinearFactory` versus a single global init mode in some older code paths.

The original paper doesn't focus on conv layer initialization; it is using global config mode, which can be uniform or normal draws, which was one possible reason for data loss in the paper plots. This repo uses targeted init in `LinearFactory` to limit shock when a new skip appears.

---

## Known limitations

1. Only `nn.Linear` hidden layers are eligible at generation time.

### Plot from regression

Graph how output of model changed compared with how many parameters the model has during adding linear residual layers.
![[Pasted image 20260510215912.png]]
