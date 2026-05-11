
## Overview

`DelLayer` removes hidden layers from a torch.fx `GraphModule`. We list all hidden modules, filter to those that pass the rules below, and that list is the delete actions the search can choose for that model.

## Generating actions

Hidden ids come from `get_all_hidden_modules` (`module_analyser.py`). For each `layer_id` we use sequential neighbours only (`module_sequential_pairs`): immediate predecessors as inputs, immediate successors as outputs.

We emit `DelLayer([layer_id])` only if every input is `nn.Linear` with the same `out_features`, every output is `nn.Linear` with the same `in_features`, and those two widths match. Otherwise we skip that id so the bypass stays a same-width linear shortcut.

## Executing actions

`DelLayer.execute` calls `delete_layer` (`model_transformations.py`). We find the `call_module` for `layer_id`, sum its `all_input_nodes` with `operator.add` if there are several, rewire every user to that sum, erase the node, drop the submodule, then `lint` and `recompile`. Non-module ops (activations, views) are not cleaned up automatically.

## Comparison with the original growingNN paper

The paper can reconnect every predecessor to every successor in a structured way. Here we sum inputs into one tensor and feed it to every successor user: simpler in FX, but not the same as every pairwise skip unless the graph already fits that story. Eligibility follows sequential adjacency and the linear checks above; see [[Model Analyser#^7a8eff]]. Rewrites live in [[Model Transformer#^f4531d]].

## Known limitations

1. Orphan-style noise: deleting only the `call_module` often leaves intermediate ops that are awkward to strip; the graph stays valid but messy. ![[Pasted image 20260506222841.png]]
2. Behaviour drifts: many deletes still change what the network does; logged runs showed strong shifts after many steps (e.g. around iteration 25). ![[Pasted image 20260506223240.png]]
3. Not everything is deletable: some hidden layers never get an action, so structure can look uneven late in a delete run (e.g. from iteration 24 in the capture). ![[Pasted image 20260510211911.png]]
