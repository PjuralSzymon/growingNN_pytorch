# Experiment 008: CIFAR-10 initial package

We keep the finished Experiment 005 package (`sequential_halving_beam` + `composed_exponential` + `val_acc` grading + 3° slope). We move that package onto CIFAR-10. Experiments 006 and 007 are unfinished, so neuron-resize stays off and the simulation set stays the default class-balanced sample.

The goal is to find which one CIFAR-specific change is required before this package can be a usable CIFAR-10 starting cell.

Script: `experiments/train_cifar10_exp008_initial_package.py`

Charts: `documentation/website/scripts/generate_experiment_008_charts.py`

Folder: `experiments/output/train_cifar10/runs/exp008_cifar10_initial_package`

Snapshot: `documentation/website/data/experiments/experiment-008-cifar10-initial-package.json`

This page is a report template. Fill tables and conclusions after the grid finishes. Charts appear once `generate_experiment_008_charts.py` has boards or a snapshot.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Package variant | `narrow`, `base`, `deep`, `epochs20`, `always`, `fixed` | Change one CIFAR knob at a time |
| Seed | `100`, `101`, `102` | Three matched seeds per variant |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | CIFAR-10 | Harder image task than MNIST |
| Planned cells | `18` | `6` variants × `3` seeds |
| Simulation algorithm | `sequential_halving_beam` | Best keep-set method from Experiment 005 |
| Look-ahead | depth `2`, beam `3` | Same as Experiment 005 |
| LR package | `composed_exponential` × logistic recovery | Best package from Experiment 004 |
| Standard cell `lr_alpha` | `0.01` | Target / peak learning rate |
| Exponential gamma | `0.98` | Base decay for `composed_exponential` |
| Accuracy metric | `val_acc` | Simulation grading |
| Score weight accuracy | `1.0` | `score_weight_acc` |
| Score weight parameter count | `0.1` | `score_weight_countw` from Experiment 005, not the old CIFAR `0.2` |
| Slope threshold | `3°` | Used by every slope variant |
| Generations | `10` | Same as Experiment 005 |
| Epochs per generation | `10` | `epochs20` uses `20` |
| Simulation time | `120 s` | Same budget as Experiment 005 |
| Simulation training epochs | `15` | Epochs inside scoring GD |
| Simulation set size | `2000` | Samples used by simulation scoring |
| Batch size | `64` | Training samples per batch |
| Starter | sequential `8/38` | `MinimalCifarNet`: two convs, one BN, one `2×2` pool, hidden linear `38` (`79060` params) |
| Augmentation | RandomCrop pad `4` + horizontal flip | CIFAR train transform in `Cifar10Data` |
| Residual-to-linear pool | average | Used if growth later inserts a residual conv before a linear |
| Neuron-resize flags | off | Experiment 006 is unfinished |

Variant meanings. Each row changes one thing from `base`.

| ID | Change from `base` | CIFAR question |
| --- | --- | --- |
| `narrow` | channels `4`, hidden `32` | Does a smaller start need growth more? |
| `base` | none. Sequential `MinimalCifarNet` channels `8`, hidden `38` | Does the MNIST package train on this CIFAR starter? |
| `deep` | channels `16`, hidden `48` | Does a wider sequential start help, or is it already too big? |
| `epochs20` | `20` epochs per generation | Is `10` epochs too short for the 3° gate on CIFAR? |
| `always` | `AlwaysSimulationScheduler` | If 3° never fires, does searching every generation help? |
| `fixed` | `NeverSimulationScheduler` | Does search help, or is this only SGD on the sequential starter? |

Run path:

```text
exp008_cifar10_initial_package/<variant_id>/<hp_folder>/seed_<seed>/
```

Smoke before the full grid:

```text
python experiments/train_cifar10_exp008_initial_package.py --variant base --seeds 100
```

If that cell takes more than two hours, cut seeds to `100` and `101` before launching the rest. Do not raise simulation time or generations in this experiment.

## Research questions

Main question: which one-factor change from the Experiment 005 package is required so GrowingNN has a usable CIFAR-10 starting cell?

Supporting checks:

1. Does `base` beat `fixed` on final validation accuracy?
2. Does the 3° slope gate run search on CIFAR, or does `always` search much more?
3. Is starter width the CIFAR knob (`narrow` `4/32`, `base` `8/38`, `deep` `16/48`), or is the `8/38` cell already enough?
4. Do `20` epochs per generation beat `10` because CIFAR climbs more slowly?
5. After an architecture action, does training accuracy recover within one generation?

## Result timeline

Progress: `0` / `18` completed (`0.0%`). Re-run this section after the grid finishes.

| Variant | Seeds done | Notes |
| --- | ---: | --- |
| `narrow` | `0` / `3` | smaller sequential starter |
| `base` | `0` / `3` | Exp 005 package on CIFAR |
| `deep` | `0` / `3` | wider sequential starter |
| `epochs20` | `0` / `3` | longer generations |
| `always` | `0` / `3` | search every generation |
| `fixed` | `0` / `3` | no architecture search |

## Why this experiment

Experiments 000 to 005 were MNIST. Experiment 005 asked for a harder dataset next. The old `train_cifar10.py` driver still uses Always-simulation, parabolic LR, and MCTS. Those settings are not the current package.

This grid does not repeat the search-algorithm comparison. It only asks what must change for CIFAR-10.

## Measurements and charts

Generate charts after runs exist:

```text
python documentation/website/scripts/generate_experiment_008_charts.py
```

### Final accuracy by variant

The first ranking is final training and validation accuracy. The later CIFAR default is the variant that is high and stable. It is not a CI lock yet. Do not call `base` best unless it also beats `fixed`.

![Final accuracy by CIFAR-10 package variant](/assets/experiments/008-final-accuracy-by-variant.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy (%) by variant. Gray markers are individual seeds.

### Does search help?

`fixed` trains the starter with no architecture actions. `base` uses the 3° slope gate. If `base` is not better on validation, search did not help this CIFAR starter under the MNIST package.

![Parameter growth by CIFAR-10 package variant](/assets/experiments/008-param-growth-by-variant.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by variant. Gray markers are individual final counts.

### Does 3° fire on CIFAR?

Count how many search calls ran and how many actions executed. `always` should search almost every generation except the last. `fixed` should search zero times. If `base` is close to `fixed`, the slope gate did not open.

![Simulations run versus actions executed](/assets/experiments/008-search-activity-by-variant.png)

> [!CAPTION] Figure 3. Mean simulation calls and executed actions by variant. Gray markers are individual seeds.

### Starter capacity

`narrow`, `base`, and `deep` keep the same gate and length. They only change sequential width (`4/32`, `8/38`, `16/48`). Residual convolution can appear only if search adds it.

![Executed action mix by CIFAR-10 package variant](/assets/experiments/008-action-composition-by-variant.png)

> [!CAPTION] Figure 4. Mean executed action counts by short label and variant.

### Recovery after architecture actions

For each executed action, compare training accuracy at the end of that generation with the next epoch and with the end of the next generation. Values are percentage points.

![Train accuracy change after architecture actions](/assets/experiments/008-post-action-recovery-by-variant.png)

> [!CAPTION] Figure 5. Mean training-accuracy change after an architecture action. Orange is the next epoch. Blue is after one recovery generation.

### Training and validation histories

Look for a late plateau, a collapse, or a long wait with no action and then a jump. `epochs20` curves are longer because each generation records 20 epochs.

![Training accuracy curves by CIFAR-10 package variant](/assets/experiments/008-training-curves.png)

> [!CAPTION] Figure 6. Training accuracy (%) over epochs for every completed seed, colored by variant.

![Validation accuracy curves by CIFAR-10 package variant](/assets/experiments/008-validation-curves.png)

> [!CAPTION] Figure 7. Validation accuracy (%) over epochs for every completed seed, colored by variant. CIFAR-10 test split is logged as val_acc.

## Grouped final results

Fill after completion.

| Variant | Mean train (%) | Mean val (%) | Mean final params | Mean simulations | Mean actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `narrow` |  |  |  |  |  |
| `base` |  |  |  |  |  |
| `deep` |  |  |  |  |  |
| `epochs20` |  |  |  |  |  |
| `always` |  |  |  |  |  |
| `fixed` |  |  |  |  |  |

## Limitations and seed effects

- Three seeds are enough for a first ranking, not for a hard reject of a close second place.
- CIFAR-10 is harder than MNIST. Final accuracy on this sequential starter is not a published CIFAR ResNet baseline.
- Simulation time stays at `120 s`. A larger net spends the same wall time and scores fewer candidates.
- Neuron-resize and new simulation-set generators are out of this grid because Experiments 006 and 007 are unfinished.
- The old CIFAR cell `ch64/hd512` is not tested. It previously crashed.
- Composed LR now interpolates from `0.001` to the current global base. Do not mix cells recorded under the older `max(0.001, global * factor)` rule.

## Conclusions

To fill after the grid:

1. State whether `base` beats `fixed` on mean validation accuracy.
2. State whether the 3° gate fires on CIFAR, or whether `always` is required.
3. State which sequential width (`narrow`, `base`, `deep`) is the CIFAR starting cell.
4. State whether `epochs20` is needed for the slope gate.
5. Name one recommended CIFAR cell for later work. Do not treat it as a CI lock.

## Next experiments

1. Copy the winning variant into a CIFAR train-ci gate the same way MNIST copied Experiments 004 and 005.
2. If 3° never fires, keep `always` only as a CIFAR exception and re-test one slope threshold later.
3. After Experiment 006 finishes, re-test neuron-resize on the winning CIFAR cell, not on MNIST `big`.
4. Do not raise simulation time to `500 s` until this 18-cell grid is done.
