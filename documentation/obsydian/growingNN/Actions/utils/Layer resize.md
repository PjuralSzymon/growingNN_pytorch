Neuron width propagation on traced `fx.GraphModule` lives in `growingnn/actions/utils/layer_resize.py`. Used by `delete_neurons.py` and `add_neurons.py`. Reads graph structure through [[Torch.fx]] `ModuleResolver`, `NodeTypeChecker`, `NodeWidthAnalyser`, and writes modules through `NodeEditor.replace_submodule` in `node_editor.py`.

Weight reprojection uses `get_reshsper` from [[Quasi identity]] inside `LinearFactory` and `ConvFactory` in `layer_Factory.py`.

---

## Public API

| Function | Role |
|----------|------|
| `can_resize_linear_output(gm, layer_id, new_width)` | Pre-check before emitting or executing a neuron action |
| `resize_layer_output(gm, layer_id, new_width)` | Replace linear output width and propagate |
| `propagate_neuron_change(gm, node, width, seen)` | Forward walk from one FX node; core propagation engine |

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

Entry point when a neuron action runs. Resizes one linear, then starts propagation.

Idea:

1. replace the target linear with a new module whose output neuron count matches the requested width (`LinearFactory.create_linear_with_rescaled_neurons`, `layer_id`, `new_width`)
2. start forward propagation from that layer's FX node (`propagate_neuron_change(gm, layer_node, new_width, seen)`)
3. recompile the graph (`gm.recompile()`); connections stay the same, only weights and module sizes change

---

## `propagate_neuron_change`

Most important function in this file. One output width changed to `width`; every downstream consumer must follow.

Idea:

1. if this graph node at this target width was already handled in an earlier pass (`key = ("p", node.name, width)` in `seen`) then stop here
2. if the node splits into multiple branches and its output width differs from the target (`NodeTypeChecker.is_fork(node)` and `NodeWidthAnalyser.node_output_width != width`) then stop here
3. for each downstream consumer of this node's output (`user in node.users`):
   3.1 if the consumer is the final graph output (`user.op == "output"`) then skip it
   3.2 if the consumer is a residual sum (`NodeTypeChecker.is_add(user)`, `nary_add` in `sum_nodes.py`) then:
       3.2.1 resize every other branch feeding that sum (`_sync_add_siblings_backward` on `inp != node`)
       3.2.2 optionally clean input widths on the branch that already changed (`_align_inputs_backward` on `node.all_input_nodes`)
       3.2.3 continue forward through the sum with the same target width (`propagate_neuron_change(gm, user, width, seen)`)
   3.3 else if the consumer is a passthrough op such as ReLU or Dropout (`NodeTypeChecker.is_passthrough`) then walk through with the same `width`
   3.4 else if the consumer is BatchNorm (`PASSTHROUGH_MODULES_TO_UPDATE`) then rescale channels (`_rescale_output_neurons`) and keep walking
   3.5 else if the consumer is a linear or conv layer (`PROPAGATION_RESIZABLE_MODULES`) then:
       3.5.1 if that layer's input width does not yet match the target (`not NodeWidthAnalyser.inputs_match_width`) then skip it
       3.5.2 rescale input connections to the target width (`_rescale_input_connections`, `width`)
       3.5.3 if the layer is square on an add path but output is still wrong (`was_square`, `NodeTypeChecker.is_add(node)`) then rescale output too (`_rescale_output_neurons`)
       3.5.4 continue propagation using that layer's new output width (`out_w = _module_output_width(updated)`)
4. record each visit in `seen` so forward, backward, and align passes do not loop on the same `node` and `width`

Rule of thumb: width is a contract. If one layer outputs N features, every direct consumer must accept N. At a residual sum, all branches must output N before the sum can move forward.

---

## `_sync_add_siblings_backward`

Called when forward propagation hits a sum. Makes sibling branches match before the sum can proceed.

Idea:

1. if this graph node at this target width was already synced (`key = ("s", node.name, width)` in `seen`) then stop here
2. if the current node is itself a residual sum (`NodeTypeChecker.is_add(node)`) then walk backward into each input (`node.all_input_nodes`)
3. else if the current node is a resizable linear or conv layer (`PROPAGATION_RESIZABLE_MODULES`) then:
   3.1 if the node is a fork outside this sum (`NodeTypeChecker.is_fork`, `at_add`) then stop here
   3.2 rescale that layer's output to the target width (`_rescale_output_neurons`, `width`)
   3.3 restart forward propagation from that node (`propagate_neuron_change(gm, node, width, seen)`)
4. else if the current node is BatchNorm or passthrough (`PASSTHROUGH_MODULES_TO_UPDATE`, `is_passthrough`) then walk further backward (`via_pass=True`)
5. else stop at forks that sit outside this sum branch

Enforces: both residual branches must output the same width before they can be added.

---

## `_align_inputs_backward`

Light backward pass before a sum. Cleans input widths on the branch that already changed.

Idea:

1. if the current node is already a direct sum input or a fork point (`node in add_node.all_input_nodes`, `NodeTypeChecker.is_fork(node)`) then stop here
2. if this graph node at this target width was already aligned (`key = ("b", node.name, width)` in `seen`) then stop here
3. first walk backward through earlier layers on this branch (`node.all_input_nodes`, recursive `_align_inputs_backward`)
4. if a layer's inputs already match the target width (`NodeWidthAnalyser.inputs_match_width`) then rescale its input connections (`_rescale_input_connections`)

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

Old GrowingNN used `scale_neurons` on custom layers and manual column slices on successor `W` matrices. R5 uses module replacement plus FX traversal. Add nodes impose the same constraint: all branch widths at a sum must stay equal after a shrink.

Chapter DOI 10.1007/978-3-031-63749-0_25 treats width change as a search move. `propagate_neuron_change` is the R5 equivalent of the old recursive `output_layers_ids` walk.

---

## Known limitations

1. Initial safe scope is Linear + BatchNorm1d + passthrough ops + `nary_add`. Conv resize helpers exist for propagation but conv neuron actions are not wired in `DelNeurons.generate_all_actions`.

2. `cat`, `view`, and attention blocks are not supported on the propagation path.

3. Shared modules used at multiple FX sites need `NodeWidthAnalyser.all_sites_match_width` before input rescale.
