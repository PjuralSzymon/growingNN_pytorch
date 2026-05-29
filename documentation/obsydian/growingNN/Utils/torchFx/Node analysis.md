File: `growingnn/utils/fx/node_analysis.py`. Per-node and per-name queries.

---

## ModuleResolver

- `get_layer_module(target, gm)` — Use this for dotted paths like `layer1.0.conv1` (`getattr(gm, "layer1.0.conv1")` is wrong).
- `find_call_module(nodes, target_name)` — first matching `call_module` node; raises with available targets if missing
- `unique_call_module_name(base, model)` — `base_0`, `base_1`, … avoiding `_modules` keys and existing FX targets. Used by all grow actions when naming new layers.

---

## NodeTypeChecker

- `is_passthrough(gm, n)` — `PASSTHROUGH_MODULES` / `PASSTHROUGH_FUNCTIONS` from `growingnn/core/config.py`
- `is_fork(n)` — more than one user
- `is_add(n)` — `operator.add` call_function

Used by `growingnn/actions/utils/layer_resize.py` when syncing widths across add branches.

---

## NodeWidthAnalyser

- `node_output_width(gm, n)` — `out_features` / `out_channels` / `num_features`, walking through passthroughs and adds
- `inputs_match_width(gm, n, w)` — all inputs share width `w`
- `all_sites_match_width(gm, module_name, w)` — every call site for that target
- `propagation_hits_unsizable(gm, start_node)` — forward walk would hit a non-`RESIZE_SAFE_MODULES` conv on a sibling branch

Used by `delete_neurons.py` and `layer_resize.py`.

---

## Generating actions

`unique_call_module_name` and `get_layer_module` run during `generate_all_actions`. Width checks gate neuron-shrink candidates in `delete_neurons.py`.
