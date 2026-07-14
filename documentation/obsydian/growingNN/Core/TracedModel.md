[[Config]]

File: `growingnn/core/traced_model.py`. Class: `TracedModel`.

This type wraps one traced `fx.GraphModule` plus the real input shape from training. It also caches graph analysis so action generation does not repeat expensive FX work on every call.

---

## Why it exists

Older code probed layer shapes with guessed tensors. `LayerShapeAnalyser.default_example_input` in `growingnn/utils/fx/graph_analysis.py` could fall back to `(1, 3, 224, 224)` when the graph had no clear hint. That size fits ImageNet-style models, not CIFAR-10 at `(1, 3, 32, 32)` or small linear chains at `(1, 4)`.

When ShapeProp ran on the wrong size, shape maps were empty or wrong. Action generators then returned no moves. Training still ran, but simulation skipped architecture search without a clear error.

`TracedModel` removes that guess. The trainer reads the first batch from the data loader and stores batch-1 shape on the wrapper. ShapeProp always uses that shape through `TracedModel.probe()` and `LayerShapeAnalyser.make_probe(gm, input_shape)`.

---

## What it does

1. holds `gm` — the live `fx.GraphModule` used for SGD and mutations
2. holds `input_shape` — tuple like `(1, 3, 32, 32)` or `(1, 4)`
3. lazily caches output shapes, input shapes, sequential pairs, dependency pairs, hidden module ids, and parameter count
4. clears all caches on `invalidate()` after the graph changes

Factory: `TracedModel.create(model, input_shape)` traces `nn.Module` when needed, or reuses an existing `GraphModule`.

---

## Where it is used

- `train_generations` in `growingnn/training/trainer.py` builds one `TracedModel` at run start from `inputs[0:1].shape`, keeps training on `traced.gm`, and passes `TracedModel` into simulation and `action.execute(traced)`
- `generate_all_actions` in [[Registry]] takes `traced: TracedModel` and passes it to every action generator
- `montecarlo_alg.py`, `greedy_alg.py`, and `random_alg.py` take `TracedModel` for search and rollouts
- each action class implements `_execute(traced)`; the base `Action.execute` in `growingnn/actions/action.py` calls `_execute` then `traced.invalidate()`

Experiment drivers still pass a plain `nn.Module` or `fx.GraphModule` into `train_generations`. They do not construct `TracedModel` themselves.

---

## Cached fields

| Field | Filled by | Used for |
|-------|-----------|----------|
| `_out_shapes`, `_in_shapes` | `shapes()` → `LayerShapeAnalyser.collect_layer_shapes` | seq/res insert eligibility, delete bypass checks |
| `_sequential_pairs` | `sequential_pairs()` → `GraphStructureQuery.module_sequential_pairs` | AddSeqLinear, AddSeqConv, AddSeqDropout |
| `_dependency_pairs` | `dependency_pairs()` → `GraphStructureQuery.module_dependency_pairs` | AddResLinear, AddResConv |
| `_hidden_modules` | `hidden_modules()` → `GraphStructureQuery.get_all_hidden_modules` | DelLayer, AddNeurons, DelNeurons |
| `_param_count` | `param_count()` → `GraphStructureQuery.get_amount_of_parameters` | trainer metrics, simulation scoring |

Public accessors: `shapes()`, `sequential_pairs()`, `dependency_pairs()`, `hidden_modules()`, `param_count()`, `probe()`, `update_shapes()`, `invalidate()`.

---

## Lifecycle

1. `TracedModel.create(model, input_shape)` — trace if needed, store shape
2. first `generate_all_actions(traced, config)` — generators call `traced.shapes()` and topology helpers; caches fill once per stable graph
3. `action.execute(traced)` — mutates `traced.gm`, then base `Action.execute` calls `traced.invalidate()`
4. next analysis call recomputes from the new graph

Simulation uses `copy.deepcopy(traced)` so rollouts do not corrupt the live wrapper. Only the action chosen by search runs on the model that continues training.

---

## Comparison with the original growingNN paper

Chapter DOI 10.1007/978-3-031-63749-0_25 describes architecture search over legal moves. The paper does not name a wrapper type or ShapeProp input size. R5 adds `TracedModel` so move generation stays tied to the dataset the run actually trains on, not a fixed 224×224 probe.

---

## Known limitations

1. `input_shape` is fixed at run start from the first training batch row. It assumes all batches share the same spatial and feature layout.

2. Code that needs shapes or topology must receive `TracedModel`, not a bare `fx.GraphModule`. There is no `TracedModel.resolve` fallback.

3. `deepcopy` of `TracedModel` duplicates the graph and caches; rollouts pay that cost per candidate in greedy and MCTS paths.

4. `invalidate()` clears all cached analysis even when a mutation is a no-op (for example neuron resize blocked by `can_resize_linear_output`).
