# Experiment 007: Simulation-set generators

We keep the best configs from the previous experiments. Experiment 004 gives the learning-rate package. Experiment 005 gives `sequential_halving_beam`. Experiment 006 is unfinished, so neuron-resize actions stay off. Only the simulation set changes.

Script: `experiments/train_mnist_exp007_simulation_sets.py`

Charts: `documentation/website/scripts/generate_experiment_007_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-007-simulation-sets.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`26` / `27` cells, `96.3%`). Untracked chart and snapshot files must be committed before raw data is deleted.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation-set generator | `protected`, `moderate_difficulty`, `kcenter`, `el2n`, `grand`, `grad_match`, `craig`, `model_drift`, `hcdc` | Compare how the simulation sample is built |
| Seed | `100`, `101`, `102` | Three matched seeds per generator |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `27` | `9` generators × `3` seeds |
| Completed cells in this refresh | `26` | `protected` seed `102` is missing |
| Simulation algorithm | `sequential_halving_beam` | Best keep-set method from Experiment 005 |
| LR package | `composed_exponential` × logistic recovery | Best package from Experiment 004 |
| Effective LR rule | `max(0.001, base_lr(epoch) * recovery_factor)` | Global exponential base times action recovery |
| Standard cell `lr_alpha` | `0.01` | Target / peak learning rate |
| Minimum LR floor | `0.001` | Hard floor on optimizer LR |
| Exponential gamma | `0.98` | Base decay for `composed_exponential` |
| Recovery warmup | logistic | Shape after an architecture action |
| Warmup length | `10` | Scheduler iterations after an action |
| Warmup steepness `k` | `10` | Logistic shape parameter |
| Accuracy metric | `val_acc` | Simulation grading via `score_by_learning.score_acc` |
| Simulation training epochs | `15` | Short GD inside scoring |
| Score weight accuracy | `1.0` | `score_weight_acc` |
| Score weight parameter count | `0.1` | `score_weight_countw` |
| Slope threshold | `3°` | `SlopeEstimationSimulationScheduler` gate |
| Generations | `10` | Same as Experiment 005 |
| Epochs per generation | `10` | Same as Experiment 005 |
| Total training epochs | `100` | `10 × 10` |
| Batch size | `64` | Training samples per batch |
| Simulation time | `120 s` | Same budget as Experiment 005 |
| Simulation set size | `2000` | Same as Experiment 005 |
| Starter | `big` (`BigAvgPoolMnistNet`) | Channels `4`, hidden `16`, same as Experiment 004 |
| Start parameter count | `420` | Same for every completed cell |
| Neuron-resize flags | off | Experiment 006 is unfinished |
| Deterministic seeding | on | `configure_deterministic_seeding()` then `seed_all(seed)` |

## Generator meanings

| ID | Class | Paper | Hypothesis |
| --- | --- | --- | --- |
| `protected` | `ProtectedSimulationSet` | none | Old class-balanced random control. |
| `moderate_difficulty` | `ModerateDifficultySimulationSet` | none for this exact method; related Paul et al. 2021 (`paul2021dataDiet`) | Middle-difficulty examples matter. |
| `kcenter` | `KCenterSimulationSet` | Sener and Savarese 2018 (`sener2018coreSet`) | Feature-space coverage matters. |
| `el2n` | `El2nSimulationSet` | Paul et al. 2021 (`paul2021dataDiet`) | Current prediction error matters. |
| `grand` | `GrandSimulationSet` | Paul et al. 2021 (`paul2021dataDiet`) | Last-layer gradient size matters. |
| `grad_match` | `GradMatchSimulationSet` | Killamsetty et al. 2021 (`pmlr-v139-killamsetty21a`) | Matching the full SGD direction matters. |
| `craig` | `CraigSimulationSet` | Mirzasoleiman et al. 2020 (`pmlr-v119-mirzasoleiman20a`) | Gradient-space coverage matters. |
| `model_drift` | `ModelDriftSimulationSet` | none | Refreshing the inner set when embeddings move matters. Default inner picker is `ProtectedSimulationSet`. |
| `hcdc` | `HcdcSimulationSet` | Ding et al. 2024 (`ding2024calibrated`); this code is a last-layer val-gradient simplification | A synthetic last-layer-gradient proxy matters. |

## Result timeline

Progress in this refresh: `26` / `27` completed (`96.3%`). Dates come from board fields `experimentStartedOn` and `lastUpdate`, not filesystem timestamps.

Board span for completed cells: `2026-08-20T22:37:05Z` to `2026-08-22T03:07:32Z`. Summed `trainingTimeElapsedSec` for the 26 completed boards is `68504 s` (about `19.0 hours`). Every completed board recorded `100` epochs on `cuda`.

| Generator | Seeds done | Notes |
| --- | ---: | --- |
| `protected` | `2` / `3` | control. Seed `102` is stuck at `50` epochs and must be rerun |
| `moderate_difficulty` | `3` / `3` | finished |
| `kcenter` | `3` / `3` | finished |
| `el2n` | `3` / `3` | finished |
| `grand` | `3` / `3` | finished |
| `grad_match` | `3` / `3` | finished |
| `craig` | `3` / `3` | finished |
| `model_drift` | `3` / `3` | finished |
| `hcdc` | `3` / `3` | finished |

## Why this experiment

Does any simulation-set generator beat `protected` on mean final validation accuracy?

Simulation scoring trains for a few epochs on a small set. Until now that set was a class-balanced random sample from `ProtectedSimulationSet`. The generators in `growingnn/simulation/simulation_sets/` try to keep a more useful subset, or a synthetic proxy, without changing search or scoring.

## Measurements and charts

Charts sort generators by mean final validation accuracy, lower on the left, higher on the right. `protected` uses two seeds. Every other generator uses three.

### Final accuracy by generator

The chart compares mean final train and validation accuracy. Gray markers are individual seeds.

![Final accuracy by simulation-set generator](/assets/experiments/007-final-accuracy-by-set.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy (%) by simulation-set generator. Gray markers are individual seeds. Sorted by mean validation, higher on the right.

| Generator | Seeds | Mean train (%) | Mean val (%) |
| --- | ---: | ---: | ---: |
| `model_drift` | `3` | `92.25` | `93.10` |
| `el2n` | `3` | `89.91` | `92.05` |
| `craig` | `3` | `89.97` | `91.36` |
| `moderate_difficulty` | `3` | `88.63` | `90.70` |
| `kcenter` | `3` | `88.36` | `90.00` |
| `protected` | `2` | `87.48` | `89.44` |
| `grad_match` | `3` | `86.91` | `88.62` |
| `grand` | `3` | `86.48` | `87.65` |
| `hcdc` | `3` | `84.02` | `87.37` |

The means sit close together. `hcdc`, `grand`, and `grad_match` are already behind and are not needed in the next grid.

### Seed stability

The chart shows every completed validation seed and the group mean.

![Seed scatter of final validation accuracy](/assets/experiments/007-seed-stability-final-val.png)

> [!CAPTION] Figure 2. Final validation accuracy (%) for each completed seed. Colored circles are seeds. Orange diamonds are means. Sorted by mean validation, higher on the right.

The seeds overlap. Size `2000` is too easy to separate the generators. The next grid should use smaller simulation sets.

### Training curves by generator

Faint lines are seeds. Bold lines are means.

![Training accuracy curves by simulation-set generator](/assets/experiments/007-training-curves.png)

> [!CAPTION] Figure 3. Training accuracy (%) over epochs. Faint lines are completed seeds. Bold lines are per-generator means.

Generation `0` still lines up inside a seed. The split appears after the first architecture action.

### Action mix by generator

The chart shows which architecture actions were executed.

![Executed action mix by simulation-set generator](/assets/experiments/007-action-composition-by-set.png)

> [!CAPTION] Figure 4. Mean executed action counts by short label and generator. Sorted by mean validation, higher on the right.

The mix is narrow. Residual conv adds dominate. Search used only a small part of the legal action list.

## Conclusions

It is hard to grade these generators at simulation set size `2000`. The means are close. The task needs to be harder.

`hcdc`, `grand`, and `grad_match` can be dropped. The next comparison can keep the other six: `protected`, `moderate_difficulty`, `kcenter`, `el2n`, `craig`, and `model_drift`.

`sequential_halving_beam` is not stable yet. Experiment 006 should finish first. Then this grid should be rerun.

## Next experiments

1. Rerun with three smaller simulation set sizes: `100`, `500`, and `1000`.
2. Keep only the six generators above.
3. Wait for Experiment 006, then rerun with a more stable simulation algorithm.
