Code: `growingnn/actions/delete_layer.py` (`DelLayer`). Uses [[Model Analyser]] `get_all_hidden_modules`, `get_input_layers`, `get_output_layers`. Uses [[Layer Analyser]] `LayerShapeAnalyser` and `LayerBridgeFinder.uniform_activation_shape`. Executes via `delete_layer` in [[Model Transformer]].

---

## Generating actions

For each hidden `layer_id` from `get_all_hidden_modules(gm)`:

1. Predecessors and successors come from sequential adjacency (`get_input_layers`, `get_output_layers`), not all transitive deps.
2. `get_common_output_shape` needs every predecessor to share the same probed output tuple (`uniform_activation_shape` on `LayerShapeAnalyser.get_layer_output_shapes`).
3. `get_common_input_shape` needs every successor to share the same probed input tuple.
4. Emit `DelLayer([layer_id])` only when `in_shape == out_shape` as full shape tuples (lines 89 to 96).

Helpers `has_same_output_shape`, `has_same_input_shape`, `get_common_output_shape`, `get_common_input_shape` live in the same file for tests (`tests/unit/actions/delete_layer_test.py`).

Shape maps are built once per `generate_all_actions` call (lines 86 to 87).

---

## Executing actions

`DelLayer.execute` calls `delete_layer(gm, layer_id)`. The FX node is removed, inputs are summed with `operator.add` when needed, users are rewired, submodule dropped, then `lint` and `recompile`. Orphan `call_function` nodes (ReLU, flatten) may remain.

---

## Comparison with the original growingNN paper

The paper allows rich reconnect patterns in theory. Here we only delete when sequential neighbours agree on activation shape from `ShapeProp`. That is stricter than old `nn.Linear` `out_features` / `in_features` checks but works for any module type that shows up in shape maps.

---

## Known limitations

1. Only hidden modules in the sense of `_is_hidden_module` are candidates.
2. Many valid deletes are skipped when shapes differ or probes fail.
3. Graph clutter after delete (BN, ReLU between modules) — see images in older notes.
4. Deleting can still change outputs a lot on ResNet runs.

---

## Related

[[Model Analyser]], [[Layer Analyser]], [[Model Transformer]], [[Dotted Module Names in torch.fx]].
