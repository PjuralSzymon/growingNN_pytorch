Neuron width updates on traced `fx.GraphModule` live in `growingnn/actions/utils/layer_resize.py`. Used by `delete_neurons.py` and `add_neurons.py`. Reads graph structure through [[Torch.fx]] `ModuleResolver`, `NodeTypeChecker`, `NodeWidthAnalyser`, and writes modules through `NodeEditor.replace_submodule` in `node_editor.py`.

Weight reprojection uses `get_reshsper` from [[Quasi identity]] inside `LinearFactory` and `ConvFactory` in `layer_Factory.py`.

---

## Public API

| Function | Role |
|----------|------|
| `can_resize_linear_output(gm, layer_id, new_width)` | Pre-check before emitting or executing a neuron action |
| `resize_layer_output(gm, layer_id, new_width)` | Replace one linear output width, then fix the whole graph |
| `fix_graph_widths(gm, align_add_to=..., pinned_head_out=...)` | Sequential sweep that repairs width mismatches |

`shrink_layer_output` and `expand_layer_output` live in `delete_neurons.py` and `add_neurons.py`; both call `resize_layer_output` after ratio math.

---

## `can_resize_linear_output`

Pre-check used before a neuron action is emitted or executed.

Idea:

1. if the target is not a linear layer (`mod`) or the new width equals the current output size (`new_width == mod.out_features`) then reject the action
2. if we are shrinking (`new_width < mod.out_features`) then:
   2.1 the new width must stay at or above the minimum allowed size (`MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL` in `growingnn/core/config.py`)
3. else if we are growing then:
   3.1 the new weight matrix must stay below the max size limit (`_within_linear_matrix_limit`, `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE`)
4. if propagation would later hit a module that cannot be resized (`NodeWidthAnalyser.propagation_hits_unsizable`) then reject the action
5. else the resize is allowed

---

## `resize_layer_output`

Entry point when a neuron action runs. Local edit first, then a whole-graph fix.

Idea:

1. snapshot the classifier Linear `out_features` (named `output` / `head` / `fc` / `classifier`, or the Linear that feeds the FX output) so class count cannot drift
2. replace the target linear with a new module whose output neuron count matches the requested width (`LinearFactory.create_linear_with_rescaled_neurons`, `layer_id`, `new_width`)
3. run `fix_graph_widths` with `align_add_to=new_width` and the pinned head size
4. recompile the graph (`gm.recompile()`); connections stay the same, only weights and module sizes change

---

## `fix_graph_widths`

Core repair loop. Does not start at the edited layer and recurse. It walks every FX node in topological order, fixes local mismatches, and repeats until a full pass makes no edits (or raises after `max_passes`).

Idea:

1. for each node in `gm.graph.nodes`:
   1.1 if a resizable module’s input width disagrees with its producer (`NodeWidthAnalyser.node_output_width`) then rescale the input (`_rescale_input_connections` / `_apply_input_resize`)
   1.2 if a residual sum (`NodeTypeChecker.is_add`, `nary_add`) has unequal input widths then pick a target width (`align_add_to` when that value is present among the inputs, else `min`) and rescale each branch’s nearest upstream output site (`_rescale_output_neurons` / `_rescale_sequential_output`); never change the pinned classifier’s `out_features` here
   1.3 if BatchNorm `num_features` disagrees with its producer then rescale (`_rescale_batch_norm`)
   1.4 if the classifier `out_features` drifted from the snapshot then restore it (`_rescale_linear_output`)
2. if a pass changed nothing then stop
3. if mismatches remain after `max_passes` then raise (fail loud; do not leave a half-fixed graph)

Rule of thumb: width is a contract. If one layer outputs N features, every direct consumer must accept N. At a residual sum, all branches must output N before the sum is valid.

---

## Rescale helpers

Idea:

1. when rescaling a layer's output then:
   1.1 reproject output weights and bias (`get_reshsper`, `_rescale_linear_output` / `_rescale_conv_output` / `_rescale_batch_norm`)
   1.2 swap in the new submodule (`NodeEditor.replace_submodule`)
2. when rescaling a layer's input then:
   2.1 only if every FX call site agrees on the target width (`NodeWidthAnalyser.all_sites_match_width`)
   2.2 reproject input-side weights (`_rescale_linear_input`, `_rescale_conv_input`)
   2.3 replace the whole submodule; never resize `Parameter` tensors in place

---

## Comparison with the original growingNN paper

Old GrowingNN used `scale_neurons` on custom layers and manual column slices on successor `W` matrices. R5 uses module replacement plus a sequential whole-graph width repair. Add nodes impose the same constraint: all branch widths at a sum must stay equal after a shrink.

Chapter DOI 10.1007/978-3-031-63749-0_25 treats width change as a search move. `fix_graph_widths` is the R5 repair step after one local width edit.

---

## Known limitations

1. Initial safe scope is Linear + BatchNorm1d + passthrough ops + `nary_add`. Conv resize helpers exist for the fix sweep but conv neuron actions are not wired in `DelNeurons.generate_all_actions`.

2. `cat`, `view`, and attention blocks are not supported on the width-fix path.

3. Shared modules used at multiple FX sites need `NodeWidthAnalyser.all_sites_match_width` before input rescale.
