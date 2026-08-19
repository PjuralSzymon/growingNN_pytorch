[[Actions]]

This action is responsible for adding layers sequentially between other layers for example for every layers connected sequentially: L1->L2->L3 we can generate actions that can create the following changes: L1-> L12-> L2 -> L3 or L1 -> L2 -> L23 -> L3 layer L1 can't connect to L3 with that type of actions. 

## Pros and cons:
### Benefits:
- Those actions don't introduce any data loss ad

Code lives in `growingnn/actions/add_seq_linear_layer.py`

It uses [[Torch.fx]]: `GraphStructureQuery.module_sequential_pairs`, `LayerShapeAnalyser`, `LayerBridgeFinder`, `ModelStructureEditor.add_new_seq_layer`, `ModuleResolver.unique_call_module_name`.

---

## Exclusion cases

For each sequential pair from `module_sequential_pairs(gm)`:

1. if `find_bridge_linear_sizes` returns `None` and `find_seq_linear_after_conv_sizes` returns `None` then skip (no square linear bridge: probed shapes missing, last dims differ because of reshape between the pair, not a linear feature dim, or conv→linear path cannot map to the successor input width)
2. if `in_features * out_features` exceeds `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE` then skip (EYE projector would be too large to allocate safely)

---

## Generating actions

`AddSeqLinearLayer.generate_all_actions(model)` probes shapes once, then walks sequential pairs.

For each pair it tries `find_bridge_linear_sizes` (linear→linear). When that fails it tries `find_seq_linear_after_conv_sizes` (conv→…→linear). Each hit builds an EYE `nn.Linear` via `LinearFactory` and appends `AddSeqLinearLayer([from_id, to_id, layer, name])`.

In the original paper for growingNN, adding sequential layers is very simple because a layer is always connected with an activation function and nodes are pretty standalone tools, but in PyTorch everything can be a module and it is hard to predict what it will be in every use case. So even if a function that goes over layers is called:
module_sequential_pairs(...)
it will only return pairs of layers placed sequentially, it is possible that between those layers there will be some activation function or something even different and custom. To work around this, we are adding the following logic:

To place a layer between l1 -> l2, we start with l2 and go backward until we find l1, and then add the new layer on a path from l2 to l1 but just before l2. This way we can handle whatever will be placed between l1 and l2, with the idea that everything between those two layers will be a part of l1, just like activation functions, not a part of l2.

## Results on complex_residual_many_widths



![[Pasted image 20260511225619.png]]


