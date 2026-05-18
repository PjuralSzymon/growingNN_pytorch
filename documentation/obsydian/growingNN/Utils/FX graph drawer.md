This page is about `growingnn/utils/fx_graph_drawer.py`. It writes Graphviz files from an `fx.GraphModule`.

### `module_weight_shape_suffix(mod)`

Lines 6 to 12. If `mod` is `nn.Linear` or a conv layer, returns a newline plus `weight` shape tuple for labels. Else returns empty string.

### `draw_torch_fx_graph(gm, output_file, fmt)`

Lines 15 to 25. Uses PyTorch `FxGraphDrawer` from `torch.fx.passes.graph_drawer`. Writes `output_file` plus extension. `fmt` is lowercased and may strip a leading dot. Supported writers come from the pydot object (`write_svg`, `write_pdf`, and so on). Raises `ValueError` if the format is unknown.

### `draw_filtered_fx_graph(gm, output_file, fmt)`

Lines 27 to 82. Builds a `graphviz.Digraph` by hand. Keeps only nodes whose `op` is `call_module` or `call_function` (line 28). Walks backward through cut nodes with `find_kept_parents` so edges still connect skipped placeholders and `output` nodes (lines 36 to 54). Labels use `gm.get_submodule(str(node.target))` for `call_module` nodes (lines 57 to 60). Calls `dot.render(output_file, format=fmt, cleanup=True)` at line 82.

### Where it is used

`tests/regression/resnet_regression_test.py` draws graphs into `testResults/regression/`. `tests/regression/utils_testing.py` does the same for a small factory model.

### Known limitations

Filtered view hides data flow through getters and placeholders; it is a sketch, not a full faithful graph. `graphviz` must be installed on the system PATH for `render` to work.

### Related

`tests/regression/resnet_regression_test.py`, `tests/regression/utils_testing.py`, `tests/regression/regression_utils.py`, `tests/run_all_test.py`.
