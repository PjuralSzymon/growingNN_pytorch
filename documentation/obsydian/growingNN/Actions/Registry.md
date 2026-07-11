[[Actions]]

File: `growingnn/actions/registry.py`. Function: `generate_all_actions(model, config)`.

This is the single entry point that builds the legal move list for one model state. Monte Carlo, greedy, and random search all call it through `RunningConfig` in `growingnn/core/config.py`. Individual action classes still expose their own `generate_all_actions`; the registry only filters and concatenates them.

---

## What it does

1. read boolean flags on `RunningConfig` (`ACTIONS_ENABLE_*`)
2. for each enabled flag call the matching action generator
3. return one flat list of `Action` instances for the current traced model

The list size changes every generation because the graph changed after the last executed action.

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

- `montecarlo_alg.get_action` — root expansion and rollouts
- `greedy_alg.get_action` — scores every action once
- `random_alg.get_action` — picks one random entry

After search picks a move, `train_generations` in `growingnn/training/trainer.py` calls `action.execute(model)` on the live model (not the simulation copy).

---

## Comparison with the original growingNN paper

The paper treats architecture moves as a search space. R5 splits move generators per action type but exposes one combined list to the search algorithm, gated at runtime by `ACTIONS_ENABLE_*` flags in config.

---

## Known limitations

1. No shared protocol type for `simulation_alg`; any module with `get_action(model, running_config)` works.

2. `can_be_infulenced` is always `False` on every action; the registry never builds compound move sequences.

3. Neuron ratios and dropout rates are fixed per flag name; new ratios need a new flag and one registry line.
