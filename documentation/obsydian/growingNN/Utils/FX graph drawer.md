File: `growingnn/utils/fx_graph_drawer.py`. Debug-only Graphviz export for an `fx.GraphModule`.

### `draw_torch_fx_graph(gm, output_file, fmt)`

Uses PyTorch `FxGraphDrawer` from `torch.fx.passes.graph_drawer`. Writes `output_file.{fmt}` (`svg`, `png`, `pdf`, …).

---

### `draw_filtered_fx_graph(gm, output_file, fmt)`

Uses PyTorch `FxGraphDrawer` but draw the simplified more clear version of a graph don't print the matrix as separate nodes only call_modules are shown in the graph 