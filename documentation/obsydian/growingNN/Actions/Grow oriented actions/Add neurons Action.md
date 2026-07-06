[[Actions]]

This page is about `growingnn/actions/add_neurons.py` and the class `AddNeurons`.

It grows the output width of one hidden `nn.Linear` layer by a ratio and propagates shapes through the FX graph. Uses `expand_layer_output`, `can_resize_linear_output`, and `resize_layer_output` in `growingnn/actions/utils/layer_resize.py`.

---

## Exclusion cases

For each hidden module from `get_all_hidden_modules(gm)`:

1. if the module is not `nn.Linear` then skip (only linear output width is implemented)
2. if `new_out == mod.out_features` after ratio rounding then skip (ratio does not change neuron count)
3. if `new_out < MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL` then skip (only when grow ratio is below 1; same guard inside `can_resize_linear_output`)
4. if `_within_linear_matrix_limit` is false then skip (`in_features * new_out` or `max(out, new_out)^2` exceeds `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE`)
5. if `NodeWidthAnalyser.propagation_hits_unsizable` is true then skip (forward propagation would hit a residual sum whose sibling branch contains modules that cannot be width-synced)

Items 2–5 are checked by `can_resize_linear_output(gm, layer_id, new_out)`.

---

## Generating actions

`AddNeurons.generate_all_actions(model, ratio=config.DEFAULT_NEURONS_GROW_RATIO)` traces the model if needed, walks hidden ids, and emits `AddNeurons([layer_id, ratio])` for each passing linear with default ratio `1.5` from `DEFAULT_NEURONS_GROW_RATIO`.

---

## Executing actions

`AddNeurons.execute` calls `expand_layer_output(model, layer_id, ratio)` which delegates to `resize_layer_output` after the same guards as shrink.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 includes width changes as architecture moves. Add neurons is the grow counterpart to delete neurons; layer-level grow remains in sequential and residual add actions.

---

## Known limitations

Same propagation limits as `layer_resize.py`. Large grow hits `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE` in `growingnn/core/config.py`.
