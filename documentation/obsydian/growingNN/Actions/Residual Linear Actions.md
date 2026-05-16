Code: `growingnn/actions/add_res_layer.py` (`AddResLayer`). Regression: `tests/regression/Adding residual layers.md` (plain path).

Uses [[Model Analyser]] `module_dependency_pairs`. Uses [[Layer Analyser]] `LayerShapeAnalyser.get_layer_output_shapes` and `LayerBridgeFinder.find_bridge_res_linear_sizes`. Executes `add_new_residual_layer` in [[Model Transformer]]. Builds layers with [[Layer Factory]] `LinearFactory.create_linear`. Names: [[Name factory]].

---

## Generating actions

`generate_all_actions(model, layer_types=...)` traces to `gm` if needed (line 27).

For each dependency pair `(layer_from_id, layer_to_id)`:

```
sizes = find_bridge_res_linear_sizes(
    out_shapes[layer_from_id],
    out_shapes[layer_to_id],
)
```

Both shapes must be rank 2. `sizes` is `(in_features, out_features)` for the skip linear.

For each `Layer_Type` in `layer_types`, append `AddResLayer([from, to, layer, name])`.

ResNet regression passes `layer_types=[Layer_Type.EYE]` only.

---

## Executing actions

`execute` → `add_new_residual_layer(model, src, dst, layer, name)` (line 17).

---

## Comparison with the original growingNN paper

DOI 10.1007/978-3-031-63749-0_25 describes architecture search with many move types. This is one move: add a parallel linear branch. Old code used `out_features` on modules; this repo uses probed shapes so lazy or custom linears are not required to be `nn.Linear` at check time.

---

## Known limitations

1. Only pairs with rank-2 output shapes on both ends get actions.
2. `module_dependency_pairs` count grows on large graphs (ResNet stress test).
3. Does not check whether `torch.add` after insert is numerically stable—only shape bridge sizes.

---

## Related

[[Residual Conv Action]], [[Layer Analyser]], [[Quasi identity]].

![[Pasted image 20260510215912.png]]
