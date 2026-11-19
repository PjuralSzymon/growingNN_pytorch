# Experiment 010: residual block actions on small image models

## Goal

Check whether enabling the new residual-block grow/shrink actions changes architecture search and training on MNIST, CIFAR-10, and CIFAR-100. Each run starts from the same small network topology, with input channels and classifier width adjusted for the dataset. The comparison arm disables only the residual-block grow and shrink flags.

**Main result:** all 12 planned runs have histories, but none selected `ResidualBlockAction`. The runs therefore do not measure the effect of applying a block mutation. Arm differences below are descriptive outcomes under two candidate-action configurations, not evidence that duplicating or removing blocks improves performance.

## Setup

| Setting | Value |
| --- | --- |
| Starter | 3×3 convolution with 8 output channels, ReLU, adaptive average pool, linear classifier |
| Starter parameters | MNIST 152; CIFAR-10 296; CIFAR-100 1,016 |
| Training | 20 generations × 10 epochs (200 epochs total) |
| Seeds | 1 and 2 per dataset and arm |
| Simulation | Monte Carlo; 8 seconds, 1 simulated epoch, 128 simulation examples |
| Optimizer settings | Cosine learning rate 0.1 → 0.001; momentum 0.9; weight decay 0.0001; gradient clipping 1.0 |
| Other actions | Enabled in both arms |
| Treatment | Residual block grow/shrink enabled vs disabled |

Experiment script: `experiments/train_cifar10_exp010_residual_block_actions.py`  
Run histories: `experiments/output/train_cifar10/runs/exp010_residual_block_actions/`  
Regenerated data snapshot: `documentation/website/data/experiments/experiment-010-residual-block-actions.json`  
Chart generator: `documentation/website/scripts/generate_experiment_010_report_data.py`

## Results

All twelve planned runs have histories: two seeds in each arm for each dataset.

| Dataset | Arm | Completed seeds | Final test accuracy by seed | Mean final test | Mean final validation | Mean final train | Final parameters by seed |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| MNIST | Enabled | 2 | 63.36%, 97.88% | 80.62% | 80.81% | 72.14% | 1,230; 10,200 |
| MNIST | Disabled | 2 | 97.52%, 97.48% | 97.50% | 97.20% | 90.34% | 10,768; 10,144 |
| CIFAR-10 | Enabled | 2 | 52.76%, 50.96% | 51.86% | 52.34% | 49.21% | 7,352; 7,792 |
| CIFAR-10 | Disabled | 2 | 59.82%, 59.92% | 59.87% | 59.35% | 55.32% | 9,248; 9,656 |
| CIFAR-100 | Enabled | 2 | 20.68%, 21.07% | 20.88% | 22.57% | 20.16% | 11,560; 8,320 |
| CIFAR-100 | Disabled | 2 | 13.85%, 20.21% | 17.03% | 17.56% | 14.84% | 6,678; 9,912 |

![Training and validation accuracy histories](/assets/experiments/010-training-histories.png)

> [!CAPTION] Figure 1. Per-seed training (solid) and validation (dashed) accuracy over 200 epochs. Blue is the enabled arm; orange is the disabled arm.

![Final official test accuracy by seed](/assets/experiments/010-final-test-accuracy.png)

> [!CAPTION] Figure 2. Final test accuracy for each completed seed. Horizontal marks show the mean of the available seeds in each arm.

![Executed action counts](/assets/experiments/010-action-counts.png)

> [!CAPTION] Figure 3. Chosen action counts summed across completed seeds for each dataset and arm. No residual-block action was chosen.

### Executed action counts

| Dataset | Arm | Actions across completed seeds | ResidualBlockAction |
| --- | --- | ---: | ---: |
| MNIST | Enabled | 38 | 0 |
| MNIST | Disabled | 38 | 0 |
| CIFAR-10 | Enabled | 38 | 0 |
| CIFAR-10 | Disabled | 38 | 0 |
| CIFAR-100 | Enabled | 38 | 0 |
| CIFAR-100 | Disabled | 38 | 0 |

Each completed run selected 19 actions, so each two-seed group contributes 38 selected actions.

## Finding

The disabled arm finished higher on MNIST by 16.88 percentage points on mean final test accuracy and on CIFAR-10 by 8.01 points. The enabled arm finished higher on CIFAR-100 by 3.85 points. MNIST’s enabled result is highly seed-sensitive: one run reached 97.88%, while the other ended at 63.36% with only 1,230 parameters. CIFAR-100 also varies substantially between its two disabled seeds (13.85% and 20.21%). These two-seed means are descriptive, not reliable estimates of treatment effects.


### Action scores and candidate generation

The run histories do **not** save the Monte Carlo simulation score assigned to each candidate. They save only the selected action and validation accuracy immediately before and after it executes. The table below therefore reports the mean immediate validation change for actions that were actually selected; it is an outcome measure, not the candidate's simulation score. Percentage changes are percentage points. Counts pool the ten completed runs.

| Selected action | Executions | Mean immediate validation change | Improved | Declined | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: |
| AddNeurons | 5 | -0.91 pp | 0 | 3 | 2 |
| AddResConvLayer | 102 | +0.01 pp | 18 | 18 | 66 |
| AddResLinearLayer | 14 | -1.05 pp | 1 | 6 | 7 |
| AddSeqConvLayer | 54 | -1.05 pp | 14 | 22 | 18 |
| AddSeqDropoutLayer | 27 | 0.00 pp | 0 | 0 | 27 |
| AddSeqLinearLayer | 17 | +0.04 pp | 7 | 4 | 6 |
| DelLayer | 3 | -2.85 pp | 0 | 3 | 0 |
| DelNeurons | 6 | -7.18 pp | 0 | 6 | 0 |
| ResidualBlockAction | 0 | — | 0 | 0 | 0 |

Candidate-generation counts are also absent from these histories. We cannot calculate how many actions of each type were generated per simulation on average, nor the mean simulation score for selected or unselected residual-block candidates. No per-run experiment-board candidate records were saved for Experiment 010. The regenerated JSON snapshot includes the observed post-action validation changes and explicitly records this missing score/count data.

The direct block-action result is **zero uses across all completed runs**. The enabled arm changed candidate generation/search and produced different choices among existing actions, but it did not execute a block duplication or removal. The report therefore cannot answer whether the new action helps. The very different MNIST seed-1 architecture and score also deserve inspection before interpreting the on/off gap.

## Next step

First add per-generation diagnostics for each action type: generated candidate count, candidate simulation score (including the chosen candidate), and whether it was selected. Then verify whether `ResidualBlockAction` is generated after the starter evolves and whether simulation can execute and grade it. Rerun a short pilot that produces a known valid block candidate and confirm those diagnostics before another full comparison. Increase seed count if a direct action-effect comparison remains the goal.

## Limitations

- Each dataset/arm cell has only two seeds, so observed means have substantial seed uncertainty.
- No completed history contains a selected `ResidualBlockAction`; no claim about block duplication/removal efficacy can be made.
- Per-candidate simulation scores and generated-action counts were not recorded, so the requested averages for received/missed candidate scores and candidate generation rates cannot be reconstructed from these artifacts.
- The experiment uses only these small image starters. It does not assess ResNet-18, Transformers, language models, or other modalities.
- Report values are derived from the final stored train/validation entries and final test entry in each `history.pt`; arm means are unweighted means over completed seeds only.
