Code path: `growingnn/actions/add_seq_conv_layer.py` (`AddSeqConvLayer`). It uses [[Model Analyser]] `module_sequential_pairs` and `get_layer_module`. It uses [[Layer Analyser]] `LayerShapeAnalyser.get_layer_output_shapes`, `get_layer_input_shapes`, and `LayerBridgeFinder.find_seq_conv_bridge_channels`. Execution calls `add_new_seq_layer` in [[Model Transformer]]. New convs use [[Layer Factory]] `ConvFactory.create_eye_conv`. Names use [[Name factory]].

## Generating actions

`generate_all_actions` walks every pair from `module_sequential_pairs(gm)`.

For each pair it reads probed shapes `s_out` and `s_in`.

It only appends an action when `find_seq_conv_bridge_channels(s_out, s_in)` is not `None`. That means both sides are rank-4 and the shapes are equal. Then it builds `ConvFactory.create_eye_conv(channels, channels, kernel_size, stride=1, padding=...)` from the source conv module.

The block that used `find_seq_conv_before_linear_sizes` and `ConvFactory.create_zero_conv_before_linear` is commented out in lines 45 to 57. It is not active.

## Executing actions

`AddSeqConvLayer.execute` calls `add_new_seq_layer(gm, layer_from_id, layer_to_id, layer, name)` like [[Sequentail Linear Actions]].

## Comparison with the original growingNN paper

In the older growingNN story, a sequential conv mutation could also place a conv block between a conv and a linear head (often with pool and flatten inside the new module).

This PyTorch port does not do that anymore. `AddSeqConvLayer` only grows conv between conv when activations already match in 4D.

Conv-to-linear sequential growth moved to [[Sequentail Linear Actions]] (`AddSeqLayer` in `growingnn/actions/add_seq_layer.py`). There a bare `nn.Linear` is inserted on the FX path after pool and flatten, so the graph becomes conv -> … -> new linear -> linear.

Reference paper chapter: https://link.springer.com/chapter/10.1007/978-3-031-63749-0_25

## Known limitations

1. No conv-between-conv-and-linear proposals (by design).
2. `get_layer_module` is still needed for `kernel_size` and `padding` on eye convs.
3. Shape probe input size may differ from your training tensor (see [[Layer Analyser]]).
4. ResNet-style heads need an existing flatten on the path; see [[Sequentail Linear Actions]].

## Sequential connections

Eye conv uses a single 1 in the centre of each channel kernel, not a full quasi-identity matrix.
