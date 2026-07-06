[[Actions]]

This page is about `growingnn/actions/delete_layer.py` and the class `DelLayer`.

It removes one hidden `call_module` from a traced model. Generation and execution use [[Torch.fx]] (`GraphStructureQuery`, `LayerShapeAnalyser`, `ModelStructureEditor.delete_layer`). Merge-branch deletes use `sum_nodes.py` (`is_merge_branch_layer`, `remove_layer_from_sums`). Eligibility uses `compute_bypass_matching` and `branch_only_bypass_compatible` in `graph_editor.py`.

---

## Exclusion cases

Only hidden modules from `get_all_hidden_modules(gm)` are candidates (`ModuleClassifier.is_hidden_module` — input-only and output-only modules are never listed).

For each hidden `layer_id`, `can_bypass_delete_layer` must return `True`. Otherwise skip:

1. if `is_merge_branch_layer` is true and the layer has no FX inputs then skip (residual side branch only feeds sums but has no upstream tensor to keep)
2. if `GraphStructureQuery.get_input_layers` is empty then skip (no editable predecessor on the trunk to bypass into)
3. if `get_output_layers` is empty and `branch_only_bypass_compatible` is false then skip (dead-end layer: not exactly one FX input, feeds a sum node, feeds a non-module op, or input activation shape ≠ successor input shape)
4. if `compute_bypass_matching` returns `None` then skip (some successor has no predecessor whose probed output shape equals that successor's probed input shape)

Use `explain_delete_layer_blockers(gm)` for `(layer_id, reason)` log lines when the list is empty.

---

## Generating actions

`DelLayer.generate_all_actions(model)` traces the model if needed, calls `LayerShapeAnalyser.collect_layer_shapes(gm)` once, then walks hidden ids (deduped with `dict.fromkeys`). Each id that passes exclusion becomes `DelLayer([layer_id])`.

---

## Executing actions

`DelLayer.execute` calls `ModelStructureEditor.delete_layer(model, layer_id)`.

`delete_layer` finds the `call_module` node, then picks one rewrite path.

Merge branch. `remove_layer_from_sums(gm, layer_node)` drops the branch tensor from every `nary_add` that consumes it. The sum is rebuilt with `_install_sum` after the latest term in topological order.

Branch-only bypass. `_rewire_branch_only_layer` replaces the layer with its single FX input in every user.

Pairwise bypass. `_rewire_layer_users` uses `compute_bypass_matching` and `_producer_before_layer` so each user gets the compatible predecessor branch, not a sum of all inputs.

After rewire, the code erases the `call_module` node and `delattr(gm, layer_id)` when the submodule is a top-level attribute.

Then `prune_unreachable_nodes(gm)` runs (`GraphConnectivity` in `graph_analysis.py`). It walks backward from the `output` node, erases every node not on the live path, and drops orphaned submodules. This removes dangling `act` nodes and dead trunk fragments left after a delete.

Finally `gm.graph.lint()` and `gm.recompile()`.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 describes architecture moves during search. Layer delete is one shrink operator. The paper can reconnect predecessors to successors in a structured way.

R5 does not sum all predecessors into one tensor for every user. It matches pairs by shape and skips only compatible branches. Residual EYE branches that only feed merges are removed from the sum, not bypassed into the trunk.

After delete, R5 enforces one connected path to the graph output via `prune_unreachable_nodes`. The old repo note about orphan FX noise is addressed for unreachable subgraphs; the live path must stay acyclic and reach `head` (or the current output module).

---

## Known limitations

1. Width-changing trunk layers (`r1_up`, `merge`, `expand`, and similar) often stay blocked because no shape-compatible bypass exists. That is expected; use neuron shrink for width change without removing the module.

2. Output drift. Delete is meant to be low shock when bypass shapes match, but logged `||Δout||` can still grow after many random deletes on a complex residual model.

3. Conv and non-linear hidden modules are not layer-deletable today. Generation only considers hidden `call_module` ids; eligibility still depends on linear shape bypass rules.

4. A live shortcut residual may remain on the only path to output after trunk deletes and prune. It is not a dangling leaf, but it is still extra structure until a later delete or resize removes it.
