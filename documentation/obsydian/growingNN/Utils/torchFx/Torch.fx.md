Package `growingnn.utils.fx` holds all torch.fx helpers for architecture mutations. Import the public API from one place:

```python
from growingnn.utils.fx import (
    ModuleClassifier, GraphStructureQuery, LayerShapeAnalyser, LayerBridgeFinder,
    ModelStructureEditor,
    ModuleResolver, NodeTypeChecker, NodeWidthAnalyser,
    NodeEditor,
)
```

Use this page as the only vault link when an action or core doc needs graph analysis or graph editing. Deeper notes live in child pages under `Utils/torchFx/` (linked below for readers already here).

---

## Read the graph

| Module | File | Role |
|--------|------|------|
| [[Graph analysis]] | `growingnn/utils/fx/graph_analysis.py` | Classify modules, list pairs and hidden layers, run `ShapeProp`, pick bridge sizes |
| [[Node analysis]] | `growingnn/utils/fx/node_analysis.py` | Resolve dotted submodule names, passthrough/fork/add checks, feature-width on nodes |

Main entry points actions call:

- `GraphStructureQuery.module_dependency_pairs` — residual add candidates (transitive over hidden modules)
- `GraphStructureQuery.module_sequential_pairs` — sequential add candidates (next editable module forward)
- `GraphStructureQuery.get_all_hidden_modules`, `get_input_layers`, `get_output_layers` — delete-layer enumeration
- `LayerShapeAnalyser.get_layer_output_shapes` / `get_layer_input_shapes` — activation shapes after `ShapeProp`
- `LayerBridgeFinder.find_*` — map probed shapes to new layer widths (linear, conv, conv-before-linear)
- `ModuleResolver.get_layer_module` — `fx.Node` or string target → live `nn.Module` via `get_submodule`
- `ModuleResolver.unique_call_module_name` — collision-free name for `gm.add_module`
- `NodeWidthAnalyser` — width checks for [[Del neurons Action]] and `layer_resize.py`

---

## Edit the graph

| Module | File | Role |
|--------|------|------|
| [[Graph editor]] | `growingnn/utils/fx/graph_editor.py` | Insert residual or sequential layers; delete a `call_module` |
| [[Node editor]] | `growingnn/utils/fx/node_editor.py` | Replace one submodule; swap one node's input edge |

Main entry points:

- `ModelStructureEditor.add_new_residual_layer` — `dst + new_layer(src)` rewrite
- `ModelStructureEditor.add_new_seq_layer` — insert on path from `src` to `dst`
- `ModelStructureEditor.delete_layer` — bypass one hidden module
- `NodeEditor.replace_submodule`, `NodeEditor.swap_node_input` — used by graph editor and neuron resize

Neuron shrink also walks the graph in `growingnn/actions/utils/layer_resize.py` using [[Node analysis]] and [[Node editor]]; that file is not part of the `utils.fx` package but uses the same API.

---

## Visualise the graph (debug)

[[FX graph drawer]] — `growingnn/utils/fx_graph_drawer.py`. Writes Graphviz SVG/PNG from a `GraphModule`. Not imported by actions.

