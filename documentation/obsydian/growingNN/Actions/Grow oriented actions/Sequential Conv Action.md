[[Actions]]

Code path: `growingnn/actions/add_seq_conv_layer.py` (`AddSeqConvLayer`). It uses [[Torch.fx]]: `GraphStructureQuery.module_sequential_pairs`, `ModuleResolver.get_layer_module`, `LayerShapeAnalyser`, `LayerBridgeFinder`, `ModelStructureEditor.add_new_seq_layer`, `ModuleResolver.unique_call_module_name`. New convs from [[Layer Factory]] `ConvFactory.create_eye_conv`.

---

## Exclusion cases

For each sequential pair from `module_sequential_pairs(gm)`:

1. if `find_seq_conv_bridge_channels` returns `None` then skip (probed output/input shapes missing, unequal, or not the same 4-D conv tensor — cannot insert a same-shape eye conv on that edge)
2. conv→linear sequential inserts are not generated (`find_seq_conv_before_linear_sizes` path is disabled in `add_seq_conv_layer.py`) then skip

---

## Generating actions

`generate_all_actions` probes shapes, walks sequential pairs, and emits `AddSeqConvLayer` when `find_seq_conv_bridge_channels` returns a channel count. Each action builds an EYE conv with the source module's `kernel_size` and `padding`.

---

## Executing actions

Same as [[Sequential Linear Actions]]: `add_new_seq_layer` on the `GraphModule`.

## Comparison with the original growingNN paper

See [[Sequential Linear Actions]] for the high-level story about activations between modules.

## Known limitations

Conv between conv and linear was removed on purpose (see section below). [[Conv to linear adapter]] is not used in this action file today.

## Sequential connections

Those layers are generated with zero weight initialization but with single 1 in the middle so we are not using the quasi-identity here because conv layers are not using matrix multiplication; those are using convolution 

## Diffrences from GrowingNN package:
I removed possiblity that can  palce a layer conv beetwen conv and linear it was very problematic and it was removed from this library 


## Results
Results on big conv model: 
![[Pasted image 20260511225255.png]]