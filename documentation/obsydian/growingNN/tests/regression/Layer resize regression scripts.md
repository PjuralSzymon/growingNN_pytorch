This note is about the GitHub issue 13 layer-resize regression scripts under `tests/regression/actions/`.

What they do. They are manual `__main__` scripts, same style as `del_neurons.py` and `adding_neurons.py`. Each run draws FX PDFs, logs Linear widths, plots `||Δout||` and param count, and prints clear FAIL lines when an invariant breaks. Shared helpers live in `layer_resize_regression_common.py`.

Why. Issue 13 says width repair after neuron resize in `growingnn/actions/utils/layer_resize.py` was unstable on nested graphs, can shrink the classification head below the class count, and cannot shrink params as far as the original GrowingNN. These scripts are the source of truth for those failure directions.

Where. Run from the repo root, for example:
`python tests/regression/actions/layer_resize_head_class_count.py --save-output true`

CLI uses `parse_regression_cli` from `tests/regression/regression_utils.py`. Default deletes `testResults/regression/` after the run unless `--save-output true`.

---

### Scripts

1. `layer_resize_head_class_count.py`
   EM-style `n_classes=47`. Scenario A uses `ModelFactory.cifar_minimal_res_conv_fork_hidden`. Scenario B uses a square head through an add (`Linear(47,47)` after a residual sum). Checks: head `out_features == 47`, and `CrossEntropyLoss` with a label equal to 46. Look for log tags `HEAD BROKEN` and `CE BROKEN`.

2. `layer_resize_nested_grow_shrink.py`
   Starts from `ModelFactory.complex_residual_many_widths`. Phase A adds residual and sequential linear layers. Phase B runs only `DelNeurons`. Looks for forward crashes and head width drift after large cascades.

3. `layer_resize_param_shrink.py`
   Grows residual and sequential linears first, then runs only `DelNeurons`. Compares end params to the pre-growth start size (soft floor ratio 2.0). Look for `PARAM FLOOR WEAK`, forward crashes on the grown graph, and `shrink stalled` while params stay high.

4. `layer_resize_mixed_conv_fork.py`
   Same fork topology as the CIFAR minimal residual that mixes `hidden`, `res_conv__0`, and `seq_linear_0`. Alternates `AddNeurons` and `DelNeurons`, prefers `seq_linear_0` on odd shrink steps. Checks head class count, CrossEntropy, and a simple fork width contract.

---

### Invariants to watch

- Classifier `out_features` must stay equal to dataset class count.
- A batch with label `n_classes - 1` must not break CrossEntropy.
- Forward `gm(x)` must keep working after every action.
- Residual branches that meet at an add must keep matching feature widths.
- Param count under shrink should be able to fall close to the start size.

---

### Comparison with the original growingNN paper

Old GrowingNN could shrink width aggressively while keeping the network trainable. These scripts make the current R5 gap visible on the same product moves: `DelNeurons` / `AddNeurons` through `resize_layer_output` and `fix_graph_widths`.

---

### Known limitations

These are regression harnesses, not pytest unit tests. A green run on one seed does not prove the redesign is done. They intentionally stress edge cases that generation currently tries to avoid with limits.
