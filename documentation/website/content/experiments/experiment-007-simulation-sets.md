# Experiment 007: Simulation-set generators

We keep the best finished GrowingNN package from Experiments 004 and 005. We change only how the simulation set is built. Experiment 006 is unfinished, so neuron-resize actions stay off.

The goal is to learn whether a smarter tiny train set for simulation scoring beats class-balanced random sampling.

Script: `experiments/train_mnist_exp007_simulation_sets.py`

Charts: `documentation/website/scripts/generate_experiment_007_charts.py`

Folder: `experiments/output/train_mnist/runs/exp007_simulation_sets`

Snapshot: `documentation/website/data/experiments/experiment-007-simulation-sets.json`

This page is a report template. Fill tables and conclusions after the grid finishes. Charts appear once `generate_experiment_007_charts.py` has boards or a snapshot.

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
| Standard cell `lr_alpha` | `0.01` | Target / peak learning rate |
| Accuracy metric | `val_acc` | Simulation grading |
| Slope threshold | `3°` | `SlopeEstimationSimulationScheduler` gate |
| Generations | `10` | Same as Experiment 005 |
| Epochs per generation | `10` | Same as Experiment 005 |
| Total training epochs | `100` | `10 × 10` |
| Simulation time | `120 s` | Same budget as Experiment 005 |
| Simulation set size | `2000` | Same as Experiment 005 |
| Starter | `big` (`BigAvgPoolMnistNet`) | Same as Experiment 004 |
| Neuron-resize flags | off | Experiment 006 is unfinished |

Generator meanings:

| ID | Class | Hypothesis |
| --- | --- | --- |
| `protected` | `ProtectedSimulationSet` | No assumption. Old class-balanced random control. |
| `moderate_difficulty` | `ModerateDifficultySimulationSet` | Middle-difficulty examples matter. |
| `kcenter` | `KCenterSimulationSet` | Feature-space coverage matters. |
| `el2n` | `El2nSimulationSet` | Current prediction error matters. |
| `grand` | `GrandSimulationSet` | Last-layer gradient size matters. |
| `grad_match` | `GradMatchSimulationSet` | Matching the full SGD direction matters. |
| `craig` | `CraigSimulationSet` | Gradient-space coverage matters. |
| `model_drift` | `ModelDriftSimulationSet` | Refreshing the set when embeddings move matters. |
| `hcdc` | `HcdcSimulationSet` | A synthetic last-layer-gradient proxy matters. |

HCDC construction time counts against the run. This grid does not compute Spearman or Kendall of action ranks. Live train and validation accuracy are the product metrics.

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

## Result timeline

Progress: `0` / `27` completed (`0.0%`). Re-run this section after the grid finishes.

| Generator | Seeds done | Notes |
| --- | ---: | --- |
| `protected` | `0` / `3` | control |
| `moderate_difficulty` | `0` / `3` | |
| `kcenter` | `0` / `3` | |
| `el2n` | `0` / `3` | |
| `grand` | `0` / `3` | |
| `grad_match` | `0` / `3` | |
| `craig` | `0` / `3` | |
| `model_drift` | `0` / `3` | wraps `protected` |
| `hcdc` | `0` / `3` | synthetic, time-capped |

## Why this experiment

Simulation scoring trains for a few epochs on a small set. Until now that set was a class-balanced random sample. The new generators in `growingnn/simulation/simulation_sets/` try to keep a more useful subset, or a synthetic proxy, without changing search or scoring.

## Measurements and charts

Generate charts after runs exist:

```text
python documentation/website/scripts/generate_experiment_007_charts.py
```

### Final accuracy by generator

![Final accuracy by simulation-set generator](/assets/experiments/007-final-accuracy-by-set.png)

> [!CAPTION] Figure 1. Mean final train and val accuracy by simulation-set generator. Gray markers are individual seeds.

### Training curves by generator

![Training accuracy curves by simulation-set generator](/assets/experiments/007-training-curves.png)

> [!CAPTION] Figure 2. Training accuracy over epochs for every completed seed, colored by generator.

### Parameter growth by generator

![Parameter growth by simulation-set generator](/assets/experiments/007-param-growth-by-set.png)

> [!CAPTION] Figure 3. Mean start and final parameter counts by generator. Gray markers are individual final counts.

### Action mix by generator

![Executed action mix by simulation-set generator](/assets/experiments/007-action-composition-by-set.png)

> [!CAPTION] Figure 4. Mean executed action counts by short label and generator.

## Grouped final results

Fill after completion.

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

## Limitations and seed effects

- Three seeds are enough for a first ranking, not for a hard reject of a close second place.
- HCDC construction time can steal search budget inside the same wall-clock run.
- Action-rank correlation on a frozen model is not measured here.
- The `big` MNIST starter may need fewer growth steps than a harder task.

## Conclusions

To fill after the grid:

1. State which generator beats `protected` on mean validation accuracy.
2. State which hypothesis that result supports.
3. State whether construction cost made a generator worse despite a better sampling idea.

## Next experiments

1. If one real-subset generator wins clearly, re-test it with a smaller `simulation_set_size`.
2. If HCDC is too slow, keep it out of default config and re-test only after a cheaper condensation cap.
3. Re-test the winner on the medium starter or a harder dataset.
