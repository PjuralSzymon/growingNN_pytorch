[[Torch.fx]]

File: `growingnn/utils/fx/graph_analysis.py`. Read-only graph queries. 

---

## ModuleClassifier

Labels `call_module` nodes.

- `is_hidden_module(node)` — middle layer, not input-only or output-only. ^7a8eff
- `is_editable_module(node, gm)` — resolved type in `EDITABLE_MODULES` from `growingnn/core/config.py`
- `is_at_least_one_hidden_module`, `is_edge_into_hidden_module` — used when walking user edges for pair lists

---

## GraphStructureQuery

Builds `gm` from `model` or uses an existing `GraphModule`, then walks `.users`.

| Method | Meaning |
|--------|---------|
| `get_all_hidden_modules(model)` | All hidden `call_module` target strings |
| `module_dependency_pairs(model)` | `(ancestor, descendant)` for residual adds; transitive over hidden |
| `module_sequential_pairs(model)` | `(a, b)` when `b` is the next editable module forward from `a` |
| `get_input_layers` / `get_output_layers` | Predecessors / successors from sequential adjacency |
| `get_amount_of_parameters(model)` | `sum(p.numel() for p in gm.parameters())` |

Example chain `l1 -> l2 -> l3` (all hidden/editable as required): dependency pairs include `(l1,l2)`, `(l1,l3)`, `(l2,l3)`; sequential pairs only `(l1,l2)` and `(l2,l3)`.

Used by `add_res_linear_layer.py`, `add_res_conv_layer.py`, `add_seq_linear_layer.py`, `add_seq_conv_layer.py`, `delete_layer.py`, `delete_neurons.py`, `add_neurons.py`.

---

## GraphConnectivity

Reachability and dangling-branch diagnostics in the same file. Used by `prune_unreachable_nodes` in [[Graph editor]].

| Method | Meaning |
|--------|---------|
| `nodes_reachable_from_output(gm)` | All FX nodes on a path backward from `output` |
| `get_output_module_id(gm)` | `call_module` target wired into `output` |
| `get_input_module_ids(gm)` | `call_module` targets fed directly by placeholders |
| `dangling_leaf_nodes(gm)` | Nodes with zero users (not `placeholder` / `output`) |
| `unreachable_module_ids(gm)` | `call_module` targets not reaching `output` |
| `live_module_ids(gm)` | `call_module` targets on the live path |
| `is_connected_to_output(gm)` | No dangling leaves and no unreachable modules |
| `explain_connectivity(gm)` | Text lines for logs |

After layer delete, a healthy graph has one input module, one output module, and `is_connected_to_output(gm) == True`.

---

## LayerShapeAnalyser

Runs `torch.fx.passes.shape_prop.ShapeProp` once per call.

- `collect_layer_shapes(gm, example)` → `(output_shapes, input_shapes)` keyed by `call_module` target; needs `example` or `input_shape` (no default 224×224 guess)
- `make_probe(gm, input_shape)` — random tensor for ShapeProp at the real batch-1 size; used by [[TracedModel]] `probe()`
- `get_layer_output_shapes`, `get_layer_input_shapes` — thin wrappers; accept optional `input_shape`
- `node_shape(node)` — reads `meta["val"]` or `tensor_meta`

Reachability from `module_dependency_pairs` does not imply equal tensor shape; conv residual filtering uses these maps in `add_res_conv_layer.py`.

---

## LayerBridgeFinder

Maps probed tuples to bridge sizes (no live `isinstance` on modules for width).

| Method | Returns | Typical caller |
|--------|---------|----------------|
| `find_bridge_linear_sizes` | `(in_f, out_f)` | `add_seq_linear_layer.py` linear→linear |
| `find_bridge_res_linear_sizes` | `(in_f, out_f)` | `add_res_linear_layer.py` |
| `find_equal_conv_output_shapes` | bool | `add_res_conv_layer.py` conv→conv |
| `find_conv_before_linear_sizes` | `(C, out)` | conv→linear residual |
| `find_seq_conv_bridge_channels` | channels | `add_seq_conv_layer.py` |
| `find_seq_linear_after_conv_sizes` | `(F, F)` | conv→linear sequential |
| `uniform_activation_shape` | one tuple or None | `delete_layer.py` shape match |

Divisors: `linear_in % channels == 0` inside `find_conv_before_linear_sizes` (conv-before-linear divisibility rule).

---

## Known limitations

1. Ranks other than 2-D linear and 4-D conv are not bridged.

2. `GraphConnectivity` checks FX reachability to `output`; it does not prove that only one residual path should remain after growth history.
