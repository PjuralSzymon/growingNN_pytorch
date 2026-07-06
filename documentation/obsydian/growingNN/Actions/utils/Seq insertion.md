Sequential grow candidate pairs. File: `growingnn/actions/utils/seq_insertion.py`. Function: `iter_seq_shape_matched_pairs(gm)`.

Uses [[Torch.fx]] `GraphStructureQuery.module_sequential_pairs` and `LayerShapeAnalyser` from `graph_analysis.py`.

---

## What it returns

Each yield is a `SeqInsertCandidate` with:

- `from_id` — predecessor editable module on a sequential edge
- `to_id` — successor editable module on the same edge
- `shape` — probed activation shape where output of `from_id` meets input of `to_id`

Only pairs where `output_shape(from_id) == input_shape(to_id)` are kept. Width-changing sequential hops are excluded here; those need bridge layers from `LayerBridgeFinder`, not plain seq insert.

---

## Idea

1. collect sequential adjacency pairs (`module_sequential_pairs`)
2. run `ShapeProp` once (`get_layer_output_shapes`, `get_layer_input_shapes`)
3. for each pair, if shapes match then yield one insert candidate

The FX path between `from_id` and `to_id` may contain activations or other ops; `ModelStructureEditor.add_new_seq_layer` inserts on the discovered path, not only on direct module-to-module edges.

---

## Comparison with the original growingNN paper

The paper assumes clean layer-to-layer edges. R5 separates pair discovery (this helper) from graph surgery (`add_new_seq_layer`) so PyTorch modules between editable endpoints still work.

---

## Known limitations

1. Shape probe uses a default example input; exotic input ranks may yield no candidates.

2. Does not check matrix size limits; sequential linear grow applies `MAX_ADD_SEQ_LAYER_WEIGHT_MATRIX_SIZE` separately in `add_seq_linear_layer.py`.
