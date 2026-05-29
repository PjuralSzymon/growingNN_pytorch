Code path: `growingnn/actions/add_seq_conv_layer.py` (`AddSeqConvLayer`). It uses [[Torch.fx]]: `GraphStructureQuery.module_sequential_pairs`, `ModuleResolver.get_layer_module`, `LayerShapeAnalyser`, `LayerBridgeFinder`, `ModelStructureEditor.add_new_seq_layer`, `ModuleResolver.unique_call_module_name`. Dotted submodule ids match [[Residual Conv Action]] and [[Del Layer Action]]. New convs from [[Layer Factory]] `ConvFactory.create_eye_conv`.

## Generating actions

`generate_all_actions` walks sequential pairs, filters when `layer_from` is a conv and `layer_to` is in a fixed tuple of pooling modules (see source lines 27 to 40 in `add_seq_conv_layer.py`).

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