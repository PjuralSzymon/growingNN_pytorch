# Experiment 007: Simulation-set generators

We keep the best finished GrowingNN package from Experiments 004 and 005. We change only how the simulation set is built. Experiment 006 is unfinished, so neuron-resize actions stay off.

The goal is to learn whether a smarter tiny train set for simulation scoring beats class-balanced random sampling.

Script: `experiments/train_mnist_exp007_simulation_sets.py`

Charts: `documentation/website/scripts/generate_experiment_007_charts.py`

Folder: `experiments/output/train_mnist/runs/exp007_simulation_sets`

Snapshot: `documentation/website/data/experiments/experiment-007-simulation-sets.json`

This page is a report template. Fill tables and conclusions after the grid finishes. Charts appear once `generate_experiment_007_charts.py` has boards or a snapshot. Untracked chart and snapshot files must be committed before raw `experiments/output/` data is deleted.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation-set generator | `protected`, `moderate_difficulty`, `kcenter`, `el2n`, `grand`, `grad_match`, `craig`, `model_drift`, `hcdc` | Compare how the simulation sample is built |
| Seed | `100`, `101`, `102` | Three matched seeds per generator |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `27` | `9` generators × `3` seeds |
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
| Neuron-resize flags | off | Experiment 006 is unfinished |
| Deterministic seeding | on | `configure_deterministic_seeding()` then `seed_all(seed)` |

How the set is built: `experiments_common._train_run` calls `RunningConfig.simulation_set_generator.generate` on the unaugmented train loader (`clean_train_loader`), the val loader, size `2000`, `seed=seed`, and `model=gm`. The ready pair is passed into `train_generations` as `sim_train_loader` / `sim_val_loader`. Live training still uses the augmented train loader. The trainer does not rebuild the set later.

Generator meanings:

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

HCDC construction time counts against the run wall clock. This grid does not compute Spearman or Kendall of action ranks. Live train and validation accuracy are the product metrics.

Run path:

```text
exp007_simulation_sets/<set_id>/<hp_folder>/seed_<seed>/
```

## Research questions

Main question: does any simulation-set generator beat `protected` on mean final validation accuracy?

Supporting checks:

1. Which hypothesis wins: difficulty, diversity, SGD direction, refresh, or synthetic data?
2. Is construction cost visible as worse actions under the same 120 s search budget?
3. Do any generators collapse the executed action mix to one family?
4. Do training curves separate early, or only after architecture actions?

## Result timeline

Progress: `0` / `27` completed (`0.0%`). Re-run this section after the grid finishes. Dates should come from board metadata, not filesystem timestamps.

| Generator | Seeds done | Notes |
| --- | ---: | --- |
| `protected` | `0` / `3` | control |
| `moderate_difficulty` | `0` / `3` | |
| `kcenter` | `0` / `3` | |
| `el2n` | `0` / `3` | |
| `grand` | `0` / `3` | |
| `grad_match` | `0` / `3` | |
| `craig` | `0` / `3` | |
| `model_drift` | `0` / `3` | wraps `protected` by default |
| `hcdc` | `0` / `3` | synthetic, time-capped |

## Why this experiment

Simulation scoring trains for a few epochs on a small set. Until now that set was a class-balanced random sample from `ProtectedSimulationSet`. The generators in `growingnn/simulation/simulation_sets/` try to keep a more useful subset, or a synthetic proxy, without changing search or scoring.

Search, LR, starter, and length stay fixed. Only `cfg.simulation_set_generator` changes.

## Measurements and charts

Generate charts after runs exist:

```text
python documentation/website/scripts/generate_experiment_007_charts.py
```

The script reads completed boards under the run folder. If raw output is absent, it falls back to the JSON snapshot.

### Final accuracy by generator

The chart compares mean final train and validation accuracy across generators. Individual seeds are markers.

![Final accuracy by simulation-set generator](/assets/experiments/007-final-accuracy-by-set.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy (%) by simulation-set generator. Gray markers are individual seeds.

### Training curves by generator

The chart shows training accuracy over epochs so early vs late separation is visible.

![Training accuracy curves by simulation-set generator](/assets/experiments/007-training-curves.png)

> [!CAPTION] Figure 2. Training accuracy (%) over epochs for every completed seed, colored by generator.

### Parameter growth by generator

The chart compares start and final parameter counts. It shows whether a generator leads search to grow the network more.

![Parameter growth by simulation-set generator](/assets/experiments/007-param-growth-by-set.png)

> [!CAPTION] Figure 3. Mean start and final parameter counts by generator. Gray markers are individual final counts.

### Action mix by generator

The chart shows which architecture actions were executed. A collapsed mix would mean search locked onto one family.

![Executed action mix by simulation-set generator](/assets/experiments/007-action-composition-by-set.png)

> [!CAPTION] Figure 4. Mean executed action counts by short label and generator.

## Grouped final results

Fill after completion. Report means over the three seeds. Accuracy in percent.

| Generator | Mean train (%) | Mean val (%) | Mean final params |
| --- | ---: | ---: | ---: |
| `protected` |  |  |  |
| `moderate_difficulty` |  |  |  |
| `kcenter` |  |  |  |
| `el2n` |  |  |  |
| `grand` |  |  |  |
| `grad_match` |  |  |  |
| `craig` |  |  |  |
| `model_drift` |  |  |  |
| `hcdc` |  |  |  |

## Training-history analysis

Fill after Figure 2 exists. Describe shapes, not the table means:

- when curves separate
- whether a drop follows an architecture action
- whether recovery follows that drop
- whether a plateau appears before repeated actions
- whether late actions keep or destroy an earlier peak

Pick one representative seed only if the three seeds agree. If they disagree, say so and show the disagreement.

## Limitations and seed effects

- Three seeds are enough for a first ranking, not for a hard reject of a close second place.
- HCDC construction time can steal search budget inside the same wall-clock run.
- Action-rank correlation on a frozen model is not measured here.
- The set is built once before training. Model-aware generators see the untrained starter `gm`, not later weights.
- The `big` MNIST starter may need fewer growth steps than a harder task.

## Conclusions

To fill after the grid:

1. State which generator beats `protected` on mean validation accuracy, or that none does.
2. State which hypothesis that result supports.
3. State whether construction cost made a generator worse despite a better sampling idea.

## Next experiments

1. If one real-subset generator wins clearly, re-test it with a smaller `simulation_set_size`.
2. If HCDC is too slow, keep it out of default config and re-test only after a cheaper condensation cap.
3. Re-test the winner on the medium starter or a harder dataset.
4. If a model-aware generator looks promising, re-test it with a rebuild after each architecture action.
