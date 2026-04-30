

### `delete_layer`

^f4531d

Removes a specified module (`layer_id`) from a `torch.fx` graph and rewires the graph to preserve connectivity.

- Collects all input nodes to the layer and all nodes that use its output
- If multiple inputs exist, combines them using `operator.add`
- Replaces the deleted layer in all downstream nodes with the new combined input
- Removes the layer node from the graph and deletes it from the module
- Recompiles the graph to reflect changes

Result: the layer is removed and its inputs are directly connected to its outputs.