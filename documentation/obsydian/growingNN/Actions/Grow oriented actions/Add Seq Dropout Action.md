[[Actions]]

This page is about `growingnn/actions/add_seq_dropout_layer.py` and the class `AddSeqDropoutLayer`.

It inserts `nn.Dropout` or `nn.Dropout2d` on a shape-matched sequential edge. Uses `iter_seq_shape_matched_pairs` from `seq_insertion.py` and `ModelStructureEditor.add_new_seq_layer` from [[Torch.fx]]. Enabled when `ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01`, `_02`, or `_05` are true on `RunningConfig` in `growingnn/core/config.py`.

---

## Exclusion cases

A sequential pair is never considered when probed output and input shapes differ (`iter_seq_shape_matched_pairs` — width-changing hops are excluded before this action runs).

1. if `_is_dropout_module` is true on either endpoint (`nn.Dropout`, `nn.Dropout2d`) then skip (do not stack dropout next to dropout)
2. if `RegularizationFactory.create_dropout` returns `None` then skip (activation rank is not 2 for vectors or 4 for `NCHW` conv tensors)

---

## Generating actions

`AddSeqDropoutLayer.generate_all_actions(model, p)` walks shape-matched sequential pairs from `iter_seq_shape_matched_pairs`.

For each passing candidate (`from_id`, `to_id`, `shape`):

1. build dropout with `RegularizationFactory.create_dropout(shape, p)` — rank 2 → `Dropout`, rank 4 → `Dropout2d`
2. pick name `seq_dropout_N` via `unique_call_module_name`
3. append `AddSeqDropoutLayer([from_id, to_id, layer, name])`

Three registry entries call the same generator with `p = 0.1`, `0.2`, or `0.5`.

---

## Executing actions

`execute` calls `ModelStructureEditor.add_new_seq_layer(model, from_id, to_id, layer, name)` — same sequential insert path as other seq grow actions.

---

## Comparison with the original growingNN paper

Dropout as an explicit grow move is an R5 extension. It changes regularization without changing tensor shape on the sequential edge.

---

## Known limitations

1. Only rank-2 and rank-4 activation shapes are supported.

2. No shrink action removes dropout yet; only grow via this action family.
