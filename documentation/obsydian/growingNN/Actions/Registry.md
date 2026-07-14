[[Actions]]

File: `growingnn/actions/registry.py`. Function: `generate_all_actions(traced, config)`.

This is the single entry point that builds the legal move list for one model state. Monte Carlo, greedy, and random search all call it through `RunningConfig` in `growingnn/core/config.py`. Individual action classes still expose their own `generate_all_actions`; the registry only filters and concatenates them.

Every call expects a [[TracedModel]] instance, not a bare `fx.GraphModule`. The wrapper carries `input_shape` for ShapeProp and reuses cached `shapes()` and topology lists across generators in one pass.

---

## What it does

1. read boolean flags on `RunningConfig` (`ACTIONS_ENABLE_*`)
2. for each enabled flag call the matching action generator with the same `traced` object
3. return one flat list of `Action` instances for the current graph state

The list size changes every generation because the graph changed after the last executed action.

During one `generate_all_actions` call, the first generator that touches `traced.shapes()` or `traced.sequential_pairs()` fills the cache. Later generators in the same call reuse that data until something invalidates it.

---

## Cache invalidation after execute

Mutations must not reuse stale shape or topology maps.

Base class `Action` in `growingnn/actions/action.py` defines:

1. `execute(traced)` — calls `_execute(traced)`, then `traced.invalidate()`
2. `_execute(traced)` — implemented per action; edits `traced.gm` only

So any `action.execute(traced)` clears `_out_shapes`, `_in_shapes`, `_sequential_pairs`, `_dependency_pairs`, `_hidden_modules`, and `_param_count` on that wrapper. The next `generate_all_actions(traced, config)` recomputes analysis on the new graph.

Call sites do not call `invalidate()` themselves. Examples: `train_generations` in `growingnn/training/trainer.py`, `greedy_alg.get_action`, and `montecarlo_alg` rollout loops only call `action.execute(traced)` or `action.execute(traced_copy)`.

---

## Enabled action families

Grow flags (`update_grow_actions` toggles all grow flags together):

| Flag | Generator | Notes |
|------|-----------|-------|
| `ACTIONS_ENABLE_ADD_RES_LAYER` | `AddResLinearLayer` | EYE and ZERO init |
| `ACTIONS_ENABLE_ADD_RES_CONV_LAYER` | `AddResConvLayer` | conv residual skip |
| `ACTIONS_ENABLE_ADD_SEQ_LAYER` | `AddSeqLinearLayer` | sequential linear insert |
| `ACTIONS_ENABLE_ADD_SEQ_CONV_LAYER` | `AddSeqConvLayer` | sequential conv insert |
| `ACTIONS_ENABLE_ADD_SEQ_DROPOUT_01/02/05` | `AddSeqDropoutLayer` | p = 0.1 / 0.2 / 0.5 |
| `ACTIONS_ENABLE_ADD_NEURONS_11/15/20` | `AddNeurons` | ratio 1.1 / 1.5 / 2.0 |

Shrink flags (`update_shrink_actions` toggles all shrink flags together):

| Flag | Generator | Notes |
|------|-----------|-------|
| `ACTIONS_ENABLE_DEL_LAYER` | `DelLayer` | hidden module delete |
| `ACTIONS_ENABLE_DEL_NEURONS_01/05/09` | `DelNeurons` | ratio 0.1 / 0.5 / 0.9 |

---

## Who calls it

- `montecarlo_alg.get_action` — root expansion and rollouts on `TracedModel` copies
- `greedy_alg.get_action` — scores every action once on deep copies
- `random_alg.get_action` — picks one random entry from the list

After search picks a move, `train_generations` in `growingnn/training/trainer.py` calls `action.execute(traced)` on the live `TracedModel` built at run start. Public API still returns `traced.gm` to experiment drivers.

---

## Comparison with the original growingNN paper

The paper treats architecture moves as a search space. R5 splits move generators per action type but exposes one combined list to the search algorithm, gated at runtime by `ACTIONS_ENABLE_*` flags in config. R5 also binds that list to dataset-shaped probes via [[TracedModel]] instead of a fixed default input size.

---

## Known limitations

1. No shared protocol type for `simulation_alg`; any module with `get_action(traced, running_config)` works.

2. `can_be_infulenced` is always `False` on every action; the registry never builds compound move sequences.

3. Neuron ratios and dropout rates are fixed per flag name; new ratios need a new flag and one registry line.

4. `invalidate()` runs after every `execute`, even when `_execute` did not change the graph.
