File: `growingnn/utils/fx/node_editor.py`. Class `NodeEditor`. Small in-place edits on one submodule or one node's inputs. Parent: [[Torch.fx]].

---

## `replace_submodule(gm, module_path, new_module)`

Splits `module_path` on the last `.`, resolves the parent with `gm.get_submodule(parent)` (or `gm` when there is no dot), then calls `add_module(leaf, new_module)`. Used when shrinking neurons: swap a smaller `nn.Linear` without rebuilding the whole graph.

---

## `swap_node_input(node, old, new)`

Replaces `old` with `new` in `node.args` and `node.kwargs`. Used by `ModelStructureEditor.add_new_seq_layer` in `graph_editor.py`.

---

## Generating actions

No enumeration here.

---

## Executing actions

`delete_neurons.py` calls `replace_submodule` after width checks in `node_analysis.py`. Sequential layer insert uses `swap_node_input` inside `graph_editor.py`.

---

## Comparison with the original growingNN paper

Neuron removal in the paper is not tied to FX. R5 uses submodule replacement plus optional graph walks in `layer_resize.py`.

---

## Known limitations

1. `replace_submodule` does not update unrelated `call_module` nodes if the same submodule is shared (unusual in traced graphs).
2. Only rewrites explicit args/kwargs on one node; does not fix downstream metadata.
