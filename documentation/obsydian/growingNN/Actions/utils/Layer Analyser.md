File: `growingnn/actions/utils/layer_analyser.py`. Two static-only classes: `LayerShapeAnalyser` (run `ShapeProp`, read shapes) and `LayerBridgeFinder` (decide bridge sizes from shapes). Not the same as [[Model Analyser]] (graph reachability).

Used by [[Sequentail Linear Actions]], [[Sequential Conv Action]], [[Residual Linear Actions]], [[Residual Conv Action]], and [[Del Layer Action]].

---

## LayerShapeAnalyser

What. Runs `torch.fx.passes.shape_prop.ShapeProp` once and builds maps keyed by `call_module` target strings.

Why. `module_dependency_pairs` says two layers are connected. That does not mean tensors match for `torch.add` or for a new bridge layer.

Methods:

- `node_shape(node)` reads `node.meta["val"]` or `tensor_meta` (lines 14 to 23).
- `default_example_input(gm)` picks `randn(1, in_features)` from the first `nn.Linear`, else `randn(1, C, 224, 224)` from the first conv, else `(1, 3, 224, 224)` (lines 26 to 40).
- `collect_layer_shapes(gm, example)` returns `(outputs, inputs)` dicts (lines 52 to 74).
- `get_layer_output_shapes(gm, example)` and `get_layer_input_shapes(gm, example)` are thin wrappers (lines 77 to 88).

Input shape for layer L is the shape of `L.args[0]` when that arg is an `fx.Node`.

---

## LayerBridgeFinder

What. Maps probed activation tuples to bridge sizes. No `isinstance` on modules for width.

Helpers:

- `linear_feature_dim(shape)` — rank 2 only; uses last dim as feature count.
- `conv_channels(shape)` — rank 4 only; uses channel dim index 1.
- `uniform_activation_shape(shapes)` — one shared tuple if all entries match; used by [[Del Layer Action]].

Bridge finders:

| Method | Returns | Used by |
|--------|---------|---------|
| `find_bridge_linear_sizes` | `(in_f, out_f)` rank-2 → rank-2 | `AddSeqLayer` linear→linear |
| `find_bridge_res_linear_sizes` | `(in_f, out_f)` from two outputs | `AddResLayer` |
| `find_equal_conv_output_shapes` | bool, equal 4D tuples | `AddResConvLayer` conv→conv |
| `find_conv_before_linear_sizes(..., for_residual=True)` | `(channels, linear_out)` | `AddResConvLayer` conv→linear residual |
| `find_seq_conv_bridge_channels` | channel count, equal 4D | `AddSeqConvLayer` conv→conv |
| `find_seq_linear_after_conv_sizes` | `(F, F)` conv 4D + linear 2D in | `AddSeqLayer` conv→linear |
| `find_seq_conv_before_linear_sizes` | `(C, C)` | commented out in seq conv action |

`find_seq_linear_after_conv_sizes` does not add pool or flatten. It sizes a plain `nn.Linear` on the existing FX path (see [[Sequentail Linear Actions]]).

Divisibility rule `linear_in % channels == 0` lives inside `find_conv_before_linear_sizes` (lines 160 to 176). Same idea as [[Conv to linear adapter]].

---

## Generating actions (where shapes matter)

`AddResConvLayer` skips conv→conv when `find_equal_conv_output_shapes` is false (e.g. ResNet `layer3` → `layer4` different H×W).

`AddSeqLayer` uses rank-2 bridges or `find_seq_linear_after_conv_sizes`.

`DelLayer` requires `in_shape == out_shape` as full tuples from neighbours (not only linear `out_features`).

---

## Executing actions

This file does not execute graph edits. Execution stays in [[Model Transformer]].

---

## Comparison with the original growingNN paper

DOI 10.1007/978-3-031-63749-0_25 does not name `ShapeProp`. The idea matches the paper: propose only moves that keep tensor math valid during search.

---

## Known limitations

1. Default probe often uses 224×224 while a forward pass may use a smaller spatial size; maps can be wrong or empty.
2. `collect_layer_shapes` runs twice if you call both getters without sharing `collect_layer_shapes` once.
3. Rank-3 or other ranks are not bridged.
4. Empty maps after failed `ShapeProp` make some actions propose nothing (fail closed for that step).
