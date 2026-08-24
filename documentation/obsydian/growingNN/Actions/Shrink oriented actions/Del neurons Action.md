[[Actions]]

This page is about `growingnn/actions/delete_neurons.py` and the class `DelNeurons`.

It shrinks the output width of one hidden `nn.Linear` layer and propagates the new shape through the FX graph. It does not erase graph nodes. Work is delegated to `growingnn/actions/utils/layer_resize.py` (`shrink_layer_output`, `resize_layer_output`, `fix_graph_widths`). Width projection uses `LinearFactory.create_linear_with_rescaled_neurons` in `layer_Factory.py` and `get_reshsper` from `quaziIdentity.py`.

---

## Exclusion cases

For each hidden module from `get_all_hidden_modules(gm)`:

1. if the module is not `nn.Linear` then skip (only linear output width is implemented)
2. if `new_out == mod.out_features` after ratio rounding then skip (ratio does not change neuron count)
3. if `new_out < MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL` then skip (shrink would leave the layer narrower than the configured floor)
4. if `_within_linear_matrix_limit` is false then skip (`in_features * new_out` or `max(out, new_out)^2` exceeds `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE`)
5. if `NodeWidthAnalyser.propagation_hits_unsizable` is true then skip (forward propagation would hit a residual sum whose sibling branch contains modules that cannot be width-synced)

Items 2–5 are checked by `can_resize_linear_output(gm, layer_id, new_out)`.

---

## Generating actions

`DelNeurons.generate_all_actions(model, ratio=config.DEFAULT_NEURONS_SHRINK_RATIO)` traces the model if needed, walks hidden ids, and emits `DelNeurons([layer_id, ratio])` for each passing linear with default ratio `0.5` from `DEFAULT_NEURONS_SHRINK_RATIO`.

---

## Executing actions

`DelNeurons.execute` reads `layer_id = params[0]` and `ratio = params[1]` (default `DEFAULT_NEURONS_SHRINK_RATIO`).

It calls `shrink_layer_output(model, layer_id, ratio)` which:

1. Resolves the `nn.Linear` submodule.
2. Computes `new_out = max(1, int(out_features * ratio))`.
3. Re-checks `can_resize_linear_output`.
4. Calls `resize_layer_output(gm, layer_id, new_out)` when allowed.

`resize_layer_output` replaces the linear with `LinearFactory.create_linear_with_rescaled_neurons`, then runs `fix_graph_widths` (sequential whole-graph sweep that syncs `nary_add` sibling branches and rescales downstream `in_features` / `out_features` / BatchNorm channels). Full repair rules are in `layer_resize.py`.

`can_be_infulenced` returns `False`; no action chains off a neuron delete in the current design.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 treats architecture search at several scales. Neuron removal is finer than whole-layer delete.

The old `Del_neurons` class reshaped custom `W` / `B` arrays and walked `output_layers_ids`. R5 keeps the same idea: shrink one layer, then update fan-out weight columns and all constrained branches. Discovery uses `node.users` and `node.all_input_nodes` from torch.fx instead of manual adjacency lists.

---

## Known limitations

1. Only `nn.Linear` output shrink is implemented in generation. Conv channel shrink is not exposed through `DelNeurons` yet.

2. Propagation skips modules that are not in `PROPAGATION_RESIZABLE_MODULES` from config. Graphs with unsupported ops on the propagation path fail `can_resize_linear_output`.

3. Grow and shrink share `can_resize_linear_output` / `resize_layer_output`; large grow is capped by `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE` in config.

4. Neuron delete does not call `prune_unreachable_nodes`. It changes module sizes in place; graph topology stays the same.
