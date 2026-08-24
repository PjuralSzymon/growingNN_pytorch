# Experiment 007: Simulation-set generators

We keep the best configs from the previous experiments. Experiment 004 gives the learning-rate package. Experiment 005 gives `sequential_halving_beam`. Experiment 006 is unfinished, so neuron-resize actions stay off.

The first pass used simulation set size `2000`. The generators were too close to grade. This grid makes the task harder. Only six generators remain. The set size is now `100`, `500`, or `1000`.

Script: `experiments/train_mnist_exp007_simulation_sets.py`

Charts: `documentation/website/scripts/generate_experiment_007_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-007-simulation-sets.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`0` / `54` cells, `0.0%`). Untracked chart and snapshot files must be committed before raw data is deleted.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation-set generator | `protected`, `moderate_difficulty`, `kcenter`, `el2n`, `craig`, `model_drift` | Compare how the simulation sample is built |
| Simulation set size | `100`, `500`, `1000` | Make scoring harder than size `2000` |
| Seed | `100`, `101`, `102` | Three matched seeds per cell |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `54` | `6` generators × `3` sizes × `3` seeds |
| Completed cells in this refresh | `0` | New grid. Size `2000` runs are not used |
| Dropped generators | `grand`, `grad_match`, `hcdc` | Weak in the size `2000` pass |
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
| `craig` | `CraigSimulationSet` | Mirzasoleiman et al. 2020 (`pmlr-v119-mirzasoleiman20a`) | Gradient-space coverage matters. |
| `model_drift` | `ModelDriftSimulationSet` | none | Refreshing the inner set when embeddings move matters. Default inner picker is `ProtectedSimulationSet`. |

## Result timeline

Progress in this refresh: `0` / `54` completed (`0.0%`). Dates will come from board fields `experimentStartedOn` and `lastUpdate`, not filesystem timestamps.

| Generator | `100` | `500` | `1000` | Notes |
| --- | ---: | ---: | ---: | --- |
| `protected` | `0` / `3` | `0` / `3` | `0` / `3` | control |
| `moderate_difficulty` | `0` / `3` | `0` / `3` | `0` / `3` | |
| `kcenter` | `0` / `3` | `0` / `3` | `0` / `3` | |
| `el2n` | `0` / `3` | `0` / `3` | `0` / `3` | |
| `craig` | `0` / `3` | `0` / `3` | `0` / `3` | |
| `model_drift` | `0` / `3` | `0` / `3` | `0` / `3` | |

## Why this experiment

Does any simulation-set generator beat `protected` on mean final validation accuracy when the simulation set is small?

Simulation scoring trains for a few epochs on a small set. Size `2000` was too easy. Sizes `100`, `500`, and `1000` should spread the generators.

## Measurements and charts

Each chart has one panel per simulation set size. Fill after runs complete.

### Final accuracy by generator

![Final accuracy by simulation-set generator](/assets/experiments/007-final-accuracy-by-set.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy (%) by generator. One panel per simulation set size. Gray markers are individual seeds.

### Seed stability

![Seed scatter of final validation accuracy](/assets/experiments/007-seed-stability-final-val.png)

> [!CAPTION] Figure 2. Final validation accuracy (%) for each completed seed. One panel per simulation set size. Colored circles are seeds. Orange diamonds are means.

### Training curves by generator

![Training accuracy curves by simulation-set generator](/assets/experiments/007-training-curves.png)

> [!CAPTION] Figure 3. Training accuracy (%) over epochs. One panel per simulation set size. Faint lines are seeds. Bold lines are per-generator means.

### Action mix by generator

![Executed action mix by simulation-set generator](/assets/experiments/007-action-composition-by-set.png)

> [!CAPTION] Figure 4. Mean executed action counts by short label and generator. One panel per simulation set size.

## Conclusions

No cells from this grid have finished yet.

## Next experiments

Wait for this `54`-cell grid. Then keep only generators that beat `protected` on the smaller sizes.
