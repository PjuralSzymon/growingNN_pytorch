Code path: `growingnn/actions/add_seq_layer.py` (`AddSeqLayer`). It uses [[Model Analyser]] `module_sequential_pairs`. It uses [[Layer Analyser]] `LayerShapeAnalyser` and `LayerBridgeFinder`. Execution calls `add_new_seq_layer` in [[Model Transformer]]. Layers come from [[Layer Factory]] `LinearFactory.create_linear` with `Layer_Type.EYE`. Names use [[Name factory]].

## What it does

It inserts a new linear module on the path between two editable modules that are sequential in the FX graph.

PyTorch graphs often have activations, pool, or flatten between two `call_module` nodes. `module_sequential_pairs` still reports conv -> linear as a pair because it skips non-editable ops.

`add_new_seq_layer` walks from the target module back to the source, then inserts the new module just before the target. Pool and flatten stay in the graph.

## Generating actions

`generate_all_actions` runs `ShapeProp` once via `get_layer_output_shapes` and `get_layer_input_shapes`.

For each sequential pair `(layer_from_id, layer_to_id)`:

1. Linear -> linear. `find_bridge_linear_sizes(s_out, s_in)` needs rank-2 shapes on both sides. It returns `(in_features, out_features)` for `LinearFactory.create_linear`.

2. Conv -> linear. `find_seq_linear_after_conv_sizes(s_out, s_in)` needs rank-4 on the conv output and rank-2 on the linear input. It returns `(F, F)` where `F` is the linear input feature size. The new layer is a plain `nn.Linear(F, F)`, not pool or flatten. The graph becomes conv -> … -> new linear -> linear.

See lines 31 to 65 in `add_seq_layer.py`.

## Executing actions

`AddSeqLayer.execute` calls `add_new_seq_layer(model, layer_from_id, layer_to_id, layer, name)`.

## Comparison with the original growingNN paper

The paper groups many sequential growth ideas under architecture search. In the old growingNN package, putting a conv adapter between conv and linear was often tied to sequential conv actions (`create_zero_conv_before_linear` style).

This repo splits that responsibility:

- [[Sequential Conv Action]] only adds conv between conv with matching 4D shapes.
- `AddSeqLayer` now owns conv -> linear growth with a linear bridge only.

That is a deliberate change from the original paper and from the older repo: sequential conv no longer fills conv-to-linear; sequential linear does.

Reference: https://link.springer.com/chapter/10.1007/978-3-031-63749-0_25 and https://github.com/PjuralSzymon/growingnn

## Known limitations

1. `find_bridge_linear_sizes` rejects 4D activations (no flatten inside the action).
2. Conv -> linear needs an existing flatten (or rank-2 path) before the target linear in the FX graph.
3. Eye linear `(F, F)` assumes the probed linear input size is correct after pool.
4. Only `Layer_Type.EYE` is used in `generate_all_actions` today (not ZERO or RANDOM).

## Results on complex_residual_many_widths

![[Pasted image 20260511225619.png]]
