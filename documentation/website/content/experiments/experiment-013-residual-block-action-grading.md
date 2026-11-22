# Experiment 013: residual-block action grading

## Question

Does enabling residual-block grow and shrink actions improve MNIST and CIFAR-10 when both arms start from the same residual block and search uses the MCTS settings selected in Experiment 010?

The comparison is actions enabled versus actions disabled. Every other architecture action stays enabled. Duplicated blocks use quasi-identity initialization, which is now the only initializer in the implementation.

## Current protocol

MNIST with block actions enabled, seed 1, was started before this 90% stop was added. Restart the script after stopping that process so every run uses the stop below. The completed initializer comparison later on this page used CIFAR-10 and CIFAR-100, 10 generations of 5 epochs, and the previous search defaults.

| Setting | Value |
| --- | --- |
| Datasets | MNIST and CIFAR-10 |
| Seeds | 1, 2, 3 for each dataset and arm |
| Starter | Conv16, batch norm, ReLU; one identity Conv16 residual block; global average pool; linear classifier |
| Training | At most 30 generations × 10 epochs (300 epochs) |
| Stop | Validation accuracy of 90%, checked at the end of a generation |
| Learning rate | Cosine decay from 0.1 to 0.001 across 300 epochs |
| Optimizer | SGD, momentum 0.9, weight decay 0.0001, gradient clipping 1.0, label smoothing 0.1 |
| Augmentation | Experiment 009 training transforms and cutout |
| Score | Validation accuracy, weight 1; parameter-count weight 0 |
| Simulation sample | 10% of the clean training split: 5,000 MNIST examples and 4,500 CIFAR-10 examples |
| Simulation time | 32 seconds, with at least 3 algorithm iteration runs |

The MCTS values are the Experiment 010 pass-2 winners. Each winner had the highest mean final validation accuracy in its one-parameter sweep.

| Parameter | Selected value |
| --- | --- |
| `MCTS_UCB1_C` | 1.0 |
| `MCTS_UCB1_USE_SQRT` | False |
| `MCTS_PROPAGATE_ROLLOUT_VALUE` | True |
| `MCTS_ROLLOUT_DEPTH` | 1 |
| `MCTS_ROLLOUT_EPOCHS` | 1 |
| `MCTS_ROLLOUT_LR` | 0.0001 |
| `SIMULATION_MIN_ALGORITHM_ITERATION_RUNS` | 3 |

The stored rollout learning rate is 0.0001. The scheduler floor raises an applied rollout rate below 0.001 up to 0.001, which is the same behavior Experiment 010 selected. The 3-iteration minimum can also extend one search past the 32-second budget.

Script: `experiments/train_cifar10_exp013_residual_block_action_grading.py`  
New runs: `experiments/output/train_cifar10/runs/exp012_residual_block_action_grading/`

```text
pipenv run python -u experiments/train_cifar10_exp013_residual_block_action_grading.py
pipenv run python -u experiments/train_cifar10_exp013_residual_block_action_grading.py mnist cifar10
```

## What the new runs should answer

Each enabled seed has a disabled seed with the same dataset and seed. Compare final test accuracy and final validation accuracy, and count how often `ResidualBlockAction` is executed.

A final-accuracy gap with zero executed block actions measures the effect of offering the action to search. It does not measure the effect of duplicating or removing a block. Experiment 012 offered the action from a starter that had no residual block, and the action was never selected. This protocol starts with one eligible block so grow and shrink candidates exist from the first search. It also drops CIFAR-100 and Tiny ImageNet: Experiment 014 already ran that wider comparison under the old search settings, and MNIST plus CIFAR-10 is enough for this question. Training stops after the generation whose final validation accuracy reaches 90%. The 30-generation budget remains for runs that stay below that accuracy.

## Completed predecessor: duplicate initialization

This section records the completed CIFAR-10 and CIFAR-100 initializer comparison. Those runs asked whether exposing the block actions changed training and which duplicate initializer received higher simulation grades. They do not use the Experiment 010 MCTS settings.

The comparison evaluated **copy**, **quasi_identity**, and **identity_residual_zero**. Quasi-identity is now the only duplicate initializer. Copy and residual-zero remain in the tables as historical evidence for that choice. The disabled controls were shared across initializers, so initializer has no effect when block actions are disabled.

| Setting | Value |
| --- | --- |
| Starter | Small convolutional classifier with one identity residual block |
| Datasets | CIFAR-10 and CIFAR-100 |
| Seeds | 1, 2, 3 |
| Training | 10 generations × 5 epochs (50 epochs per completed run) |
| Search | Monte Carlo, 4 seconds, one rollout epoch, 256 sampled examples |
| Score | Validation accuracy; parameter-count weight 0 |
| Runs | 30 completed histories: 18 enabled (3 initializers × 2 datasets × 3 seeds) and 12 shared disabled controls |

Historical artifacts: `experiments/output/train_cifar10/runs/exp011_residual_block_action_grading/`  
Predecessor charts: `documentation/website/scripts/generate_experiment_013_grading_charts.py` and `documentation/website/scripts/generate_experiment_013_grading_history_charts.py`

The current runner does not repeat this initializer comparison. Completed attempts are included below; archived failed attempts are excluded.

## Results by seed

Final values after epoch 50:

| Dataset | Initializer / arm | Seed 1 test | Seed 2 test | Seed 3 test | Mean test ± sample SD | Mean validation | Mean training | Mean final parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | Copy enabled | 62.81% | 62.56% | 58.24% | 61.20% ± 2.57 pp | 60.77% | 58.04% | 13,018 |
| CIFAR-10 | Quasi-identity enabled | 60.05% | 64.29% | 62.38% | 62.24% ± 2.12 pp | 62.49% | 59.50% | 14,343 |
| CIFAR-10 | Residual-zero enabled | 62.22% | 58.18% | 43.58% | 54.66% ± 9.81 pp | 54.47% | 49.21% | 11,914 |
| CIFAR-10 | Actions disabled | 61.33% | 64.35% | 46.51% | 57.40% ± 9.55 pp | 57.48% | 54.67% | 16,099 |
| CIFAR-100 | Copy enabled | 28.93% | 23.31% | 18.18% | 23.47% ± 5.38 pp | 23.90% | 19.43% | 19,305 |
| CIFAR-100 | Quasi-identity enabled | 22.15% | 25.14% | 27.88% | 25.06% ± 2.87 pp | 25.35% | 22.42% | 16,585 |
| CIFAR-100 | Residual-zero enabled | 22.43% | 28.52% | 25.60% | 25.52% ± 3.05 pp | 26.17% | 23.98% | 16,964 |
| CIFAR-100 | Actions disabled | 26.05% | 24.03% | 28.47% | 26.18% ± 2.22 pp | 26.47% | 24.52% | 16,084 |

The per-seed results and run-level histories are preserved in the output folders. Mean validation and training values are averages of each run’s final epoch across its three seeds. Parameter means average final parameter counts across seeds.

## Training history and model size

![Training and validation histories by initialization](/assets/experiments/013-training-histories.png)

> [!CAPTION] Figure 1. Per-epoch train (solid) and validation (dashed) accuracy by dataset and initializer. Enabled runs are colored by seed; gray lines are the shared disabled controls. Vertical dotted lines mark generation boundaries.

![Parameter history by initialization](/assets/experiments/013-parameter-histories.png)

> [!CAPTION] Figure 2. Parameter count after each generation. Each trace is one completed run; gray traces are the shared disabled controls.

## Final accuracy comparison

![Final test accuracy by initialization and dataset](/assets/experiments/013-initialization-test-comparison.png)

> [!CAPTION] Figure 3. Final test accuracy for each seed. Black horizontal marks show means; disabled control results are repeated as the reference for each dataset.

CIFAR-10 quasi-identity has the highest enabled mean test accuracy (62.24%), 0.64 percentage points above copy. CIFAR-100 copy has the lowest enabled mean (23.47%); quasi-identity and residual-zero are similar at 25.06% and 25.52%. These differences are small relative to seed variation and do not identify a reliable best initializer.

## Which settings raised residual-block candidate grades?

In the historical comparison, identity-style initialization was associated with higher **duplicate** grades than copied weights. The current implementation fixes this setting to quasi-identity. The simulation scoring weights were held fixed across runs: validation accuracy had weight 1.0, while parameter count, loss, and time had weight 0. The recorded candidate score is therefore an absolute validation-accuracy rollout value; it does not measure improvement over the unmodified model and it does not penalize added parameters.

| Dataset | Copy: mean score (n) | Quasi-identity: mean score (n) | Residual-zero: mean score (n) |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 0.2925 (20) | 0.4543 (27) | 0.4523 (29) |
| CIFAR-100 | 0.0688 (24) | 0.1741 (28) | 0.1894 (24) |

![Residual-block duplicate and remove candidate scores by initializer](/assets/experiments/013-block-grades-by-initialization.png)

> [!CAPTION] Figure 4. Distribution of residual-block candidate rollout scores, grouped by dataset and duplicate initializer. Green triangles mark means; orange lines mark medians. Samples pool candidate appearances across seeds and generations; a candidate received one or two visits.

**Decision: quasi-identity is now the only duplicate initializer in the implementation.** Figure 4 shows that its mean duplicate grade is nearly the same as residual-zero on CIFAR-10 (0.4543 vs. 0.4523), and somewhat lower on CIFAR-100 (0.1741 vs. 0.1894). Both identity-style modes score above copied weights on both datasets. Quasi-identity is already verified and tested in this project, so the close, mixed grade comparison supports standardizing on it; this is a practical project choice, not evidence that it is statistically superior on every dataset.

A useful like-for-like architecture comparison is CIFAR-10, seed 2, generation 1. Each run had 3,242 parameters and had selected the same preceding sequential-linear action. The duplicate candidate scored **0.2833** with copy initialization, versus **0.4333** with either quasi-identity or residual-zero. Its raw-score rank among 26 candidates shifted from 14th to 3rd or 4th. The pre-simulation validation accuracies were not identical (44.94%, 46.36%, and 48.38%), however, so this is not an exact same-checkpoint initializer ablation.

The initializer effect is not a universal ranking. Across all candidate appearances, duplicate means remain below some other action types: on CIFAR-10, for example, Add Neurons averaged 0.4919 in quasi-identity runs; on CIFAR-100, Add Res Linear averaged 0.2150 in residual-zero runs. The evidence points to identity-style initialization raising duplicate scores relative to copied weights, but it does not show that block duplication generally earns the highest grade.

### What the simulation score and selection mean

These predecessor runs use a four-second MCTS search, 256 sampled examples, one epoch per scoring pass, and a one-epoch candidate expansion. MCTS then follows up to two random rollout actions (one epoch each) before scoring validation accuracy. Thus the candidate’s recorded rollout value includes downstream rollout steps, not only the isolated effect of the block action. Each root candidate received one or two visits, so these estimates are noisy.

MCTS selects using UCB rather than raw score alone. Those runs used the legacy formula with `MCTS_UCB1_C=2`. The board’s `ucbScore` and `scoreChosen` are not the same as the raw candidate grade. This mattered on CIFAR-100: the selected duplicate scored 0.21, tied with an Add Seq Conv candidate, but its UCB value was higher because it had fewer visits. On CIFAR-10, the selected quasi-identity and residual-zero duplicates each had the top raw score in their seed-1 generation-1 searches.

## Action selection and measured action effects

Block actions were executed in three of the 18 enabled runs:

| Dataset | Initializer | Seed | Generation | Selected action |
| --- | --- | ---: | ---: | --- |
| CIFAR-10 | Quasi-identity | 1 | 1 | Duplicate |
| CIFAR-10 | Residual-zero | 1 | 1 | Duplicate |
| CIFAR-100 | Residual-zero | 1 | 2 | Duplicate |

No residual-block removal was selected. The original copy runs selected no block action, and the other three-seed initializer arms selected none. Thus the final outcomes compare training with the action available to search; only three runs actually applied a block duplicate.

![Training histories and before/after generation measurements for each selected residual-block duplicate](/assets/experiments/013-block-duplicate-impact.png)

> [!CAPTION] Figure 5. Each panel marks the selected duplicate and highlights the following five-epoch generation. The lower panels compare end-of-generation training and validation accuracy and show the parameter-count change. The metric differences are descriptive; continued training and other factors can affect them.

Each generation in this experiment lasts **5 epochs**. The two CIFAR-10 duplicates were selected at global epoch 10, so the vertical marker is between epochs 10 and 11 and the shaded interval covers epochs 11–15. The CIFAR-100 duplicate was selected at epoch 15; its shaded interval covers epochs 16–20. The shaded region extends to the next half-epoch boundary because plotted values are centered on integer epoch positions.

### Model structure around each duplicate

The board saved simplified FX graphs immediately before and after each selected duplicate. The after graph is the resulting model at the end of the following generation. New module names ending in `_block_0` identify the copied residual branch.

![CIFAR-10 quasi-identity seed 1 structure before and after block duplication](/assets/experiments/013-block-duplicate-structure-cifar10-quasi_identity-seed1.png)

> [!CAPTION] Figure 6. CIFAR-10, quasi-identity, seed 1. The duplicate was selected after generation 1; generation 2 contains the copied branch.

![CIFAR-10 residual-zero seed 1 structure before and after block duplication](/assets/experiments/013-block-duplicate-structure-cifar10-identity_residual_zero-seed1.png)

> [!CAPTION] Figure 7. CIFAR-10, residual-zero, seed 1. The duplicate was selected after generation 1; generation 2 contains the copied branch.

![CIFAR-100 residual-zero seed 1 structure before and after block duplication](/assets/experiments/013-block-duplicate-structure-cifar100-identity_residual_zero-seed1.png)

> [!CAPTION] Figure 8. CIFAR-100, residual-zero, seed 1. The duplicate was selected after generation 2; generation 3 contains the copied branch.

The three duplicates were followed by improved training accuracy in the next generation, but validation did not improve consistently. CIFAR-10 quasi-identity changed train/validation accuracy by **+3.83/−4.02 pp**; CIFAR-10 residual-zero by **+2.76/−0.14 pp**; and CIFAR-100 residual-zero by **+1.28/−0.10 pp**. The recorded parameter count rose by **2,336** in each case. These before/after measurements cover the next five training epochs and show association, not the duplicate's isolated causal effect.

![Accuracy changes over the generation following selected actions](/assets/experiments/013-action-effects.png)

> [!CAPTION] Figure 9. Train and validation accuracy changes from the end of the action generation to the end of the next generation, grouped by selected action type. Dots are individual actions and bars show action-type means. These windows include continued training and later effects, so they are descriptive associations, not causal estimates.

## Interpretation

The results do not establish that residual-block actions improve training. The original copy-enabled arm exceeded disabled on CIFAR-10 by 3.81 pp, but trailed it on CIFAR-100 by 2.71 pp; no block action was applied in those copy runs. Across the new initializer arms, quasi-identity was strongest on CIFAR-10 while residual-zero was slightly higher on CIFAR-100. The project will use quasi-identity as its default duplicate initializer because the grade difference is small and mixed across datasets, and quasi-identity is already verified and tested. The three-seed sample and variable outcomes do not establish that it is universally superior.

This report updates the earlier interpretation: the action was not merely generated but was selected and applied in three follow-up runs. Those runs show that the action can participate in training under the new modes, but they do not isolate the effect of a duplicate from seed, initializer, and subsequent architecture changes.

## Limitations of the predecessor

- Only one small identity-residual starter and CIFAR-10 plus CIFAR-100 were evaluated.
- Three seeds per setting are not enough to resolve the observed seed variation.
- The initializer groups are not paired checkpoint experiments; their training trajectories and action choices can diverge.
- The current protocol above replaces the proposed checkpoint-copy follow-up. It compares block actions enabled and disabled on MNIST and CIFAR-10, for at most 30 generations, and stops when validation accuracy reaches 90%. Search uses the Experiment 010 MCTS settings.


