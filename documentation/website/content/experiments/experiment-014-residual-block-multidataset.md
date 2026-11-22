# Experiment 014: residual-block actions across image datasets

## Status

This page keeps an interim four-dataset run. The active comparison is [Experiment 013](/experiments/experiment-013-residual-block-action-grading/), which uses MNIST and CIFAR-10, 30 generations, and the MCTS settings selected in Experiment 010. The runs below used the previous search defaults: 8 seconds, 128 simulation examples, and a 90% validation stop.

**Interim report, refreshed 2026-11-19 21:27 UTC.** Of 24 planned runs, 19 are complete, one is marked running, and four have not started. The active run is Tiny ImageNet, block actions enabled, seed 2: its board contains 203 epoch records out of 300. Its latest saved board update is 21:00 UTC, with 18.49% validation accuracy. The other unstarted runs are Tiny ImageNet enabled seed 3 and all three disabled seeds.

The on/off comparison is complete for MNIST, CIFAR-10, and CIFAR-100. Tiny ImageNet has one completed enabled seed and one partial enabled seed; there is no completed disabled control yet.

## Question and design

Does allowing residual-block grow/shrink actions improve training when both arms start from the same eligible residual architecture? Each enabled run is matched by dataset and seed with a run that disables only the residual-block actions. Other architecture actions remain enabled.

| Setting | Value |
| --- | --- |
| Datasets | MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet (200 classes, not full ImageNet) |
| Starter | Conv16 + batch norm + ReLU; one Conv16 residual block; global average pool; linear classifier |
| Seeds | 1, 2, 3 per dataset and arm |
| Maximum training | 30 generations × 10 epochs (300 epochs) |
| Stop condition | Validation accuracy ≥90%; otherwise train to 300 epochs |
| Learning rate | Cosine decay 0.1 → 0.001 across the maximum budget |
| Augmentation | Experiment 009 transforms and cutout on training data |
| Search | Monte Carlo, configured for 8 seconds, one simulation epoch, 128 sampled examples |
| Simulation score | Validation accuracy; parameter-count weight 0 |
| Optimizer | SGD, momentum 0.9, weight decay 0.0001, gradient clipping 1.0 |
| Duplicate initialization | Quasi-identity |

Script: **experiments/train_cifar10_exp014_residual_block_multidataset.py**  
Run data: **experiments/output/train_cifar10/runs/exp012_residual_block_multidataset/**  
Chart/snapshot generator: **documentation/website/scripts/generate_experiment_014_report.py**

The output folder keeps the name `exp012_residual_block_multidataset` so the partial histories stay attached to this page.

The completed Tiny ImageNet enabled seed 1 took 10.52 hours for 300 epochs. Its 29 recorded MCTS searches averaged 174.6 seconds each (median 153.9 seconds, maximum 383.0 seconds), despite the configured 8-second budget. Search time is therefore a major part of the run time and the configured duration is not a wall-clock cap in these runs.

## Results by completed run

Means below use completed runs only. Train, validation, and test are final accuracies; parameters are the final parameter count. Tiny ImageNet means currently represent only enabled seed 1.

| Dataset | Arm | Completed seeds | Mean train | Mean validation | Mean test | Mean parameters |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| MNIST | Enabled | 1, 2, 3 | 79.51% | 90.72% | 91.21% | 2,773 |
| MNIST | Disabled | 1, 2, 3 | 79.09% | 91.97% | 92.50% | 4,229 |
| CIFAR-10 | Enabled | 1, 2, 3 | 57.65% | 62.95% | 63.92% | 52,340 |
| CIFAR-10 | Disabled | 1, 2, 3 | 70.07% | 72.98% | 72.85% | 47,903 |
| CIFAR-100 | Enabled | 1, 2, 3 | 28.86% | 32.43% | 31.78% | 56,601 |
| CIFAR-100 | Disabled | 1, 2, 3 | 33.45% | 35.51% | 34.44% | 57,788 |
| Tiny ImageNet | Enabled | 1 | 21.04% | 21.59% | 22.02% | 46,056 |
| Tiny ImageNet | Disabled | — | — | — | — | — |

### Per-seed results and partial run

| Dataset | Arm | Seed | Status | Train | Validation | Test | Parameters | Epochs |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | Enabled | 1 | Complete | 79.84% | 90.63% | 91.18% | 2,682 | 10 |
| MNIST | Enabled | 2 | Complete | 78.78% | 91.00% | 91.21% | 2,954 | 13 |
| MNIST | Enabled | 3 | Complete | 79.92% | 90.53% | 91.24% | 2,682 | 9 |
| MNIST | Disabled | 1 | Complete | 79.11% | 92.30% | 93.10% | 5,002 | 16 |
| MNIST | Disabled | 2 | Complete | 73.18% | 91.08% | 91.42% | 2,682 | 17 |
| MNIST | Disabled | 3 | Complete | 84.98% | 92.54% | 92.98% | 5,002 | 13 |
| CIFAR-10 | Enabled | 1 | Complete | 67.17% | 67.76% | 69.07% | 47,482 | 300 |
| CIFAR-10 | Enabled | 2 | Complete | 56.65% | 58.86% | 59.53% | 54,490 | 300 |
| CIFAR-10 | Enabled | 3 | Complete | 49.14% | 62.24% | 63.17% | 55,049 | 300 |
| CIFAR-10 | Disabled | 1 | Complete | 67.47% | 68.88% | 69.24% | 33,964 | 300 |
| CIFAR-10 | Disabled | 2 | Complete | 71.32% | 75.84% | 75.36% | 48,646 | 300 |
| CIFAR-10 | Disabled | 3 | Complete | 71.42% | 74.22% | 73.94% | 61,098 | 300 |
| CIFAR-100 | Enabled | 1 | Complete | 22.99% | 25.74% | 25.27% | 51,028 | 300 |
| CIFAR-100 | Enabled | 2 | Complete | 29.45% | 34.36% | 33.49% | 60,276 | 300 |
| CIFAR-100 | Enabled | 3 | Complete | 34.14% | 37.18% | 36.59% | 58,500 | 300 |
| CIFAR-100 | Disabled | 1 | Complete | 33.23% | 36.12% | 34.76% | 56,555 | 300 |
| CIFAR-100 | Disabled | 2 | Complete | 31.49% | 33.98% | 32.55% | 60,820 | 300 |
| CIFAR-100 | Disabled | 3 | Complete | 35.65% | 36.44% | 36.00% | 55,988 | 300 |
| Tiny ImageNet | Enabled | 1 | Complete | 21.04% | 21.59% | 22.02% | 46,056 | 300 |
| Tiny ImageNet | Enabled | 2 | Running snapshot | 18.39% | 18.49% | — | 30,296 | 203 records |

All six MNIST runs reached the 90% validation stop between epochs 9 and 17. The CIFAR runs and completed Tiny ImageNet seed 1 used the full 300 epochs. The partial Tiny ImageNet seed 2 is far below the stop threshold.

## Training history and model size

![Per-epoch train and validation accuracy for all runs](/assets/experiments/014-training-histories.png)

> [!CAPTION] Figure 1. Solid lines show training accuracy and dashed lines show validation accuracy. Color identifies seed; × marks the latest point in the partial Tiny ImageNet enabled seed 2 run. The dotted line is the 90% validation stop.

![Parameter count across completed and partial runs](/assets/experiments/014-parameter-histories.png)

> [!CAPTION] Figure 2. Parameter count by epoch. Solid traces are enabled arms and dashed traces are disabled arms; color identifies seed. Tiny ImageNet currently has enabled seeds 1 and 2 only.

The action-enabled CIFAR-10 runs ended with 4,437 more parameters on average than controls, while their test accuracy mean was 8.93 percentage points lower. CIFAR-100 enabled runs averaged 1,187 fewer parameters and 2.66 points lower test accuracy. These differences are descriptive: no residual-block action was executed in the completed MNIST or CIFAR runs.

## Final test accuracy

![Final test accuracy by seed for completed runs](/assets/experiments/014-final-test-accuracy.png)

> [!CAPTION] Figure 3. Each dot is one completed seed and each horizontal line is the arm mean. Tiny ImageNet has one completed enabled seed and no completed disabled control; its partial seed 2 is excluded.

## Residual-block candidates and executions

The action was offered in simulation on the enabled arms. Candidate scores below are recorded MCTS rollout scores, not improvements over the unchanged model. The chosen column counts simulation records in which a residual-block candidate was selected; the execution column is confirmed from the run action history.

| Dataset | Operation | Candidate appearances | Mean score | Chosen | Executed so far |
| --- | --- | ---: | ---: | ---: | ---: |
| MNIST | Duplicate | 1 | 0.88 | 0 | 0 |
| MNIST | Remove | 1 | 0.10 | 0 | 0 |
| CIFAR-10 | Duplicate | 21 | 0.44 | 0 | 0 |
| CIFAR-10 | Remove | 21 | 0.27 | 0 | 0 |
| CIFAR-100 | Duplicate | 42 | 0.23 | 0 | 0 |
| CIFAR-100 | Remove | 42 | 0.12 | 0 | 0 |
| Tiny ImageNet | Duplicate | 18 | 0.11 | 2 | 2 |
| Tiny ImageNet | Remove | 18 | 0.06 | 0 | 0 |

![MCTS scores for residual-block candidates](/assets/experiments/014-block-candidate-scores.png)

> [!CAPTION] Figure 4. Points show candidate rollout scores saved in the board simulation JSON; horizontal bars are means. The two Tiny ImageNet duplicate selections correspond to enabled seed 1 and the ongoing enabled seed 2.

Both Tiny ImageNet enabled runs selected a duplicate after generation 0. In seed 1, validation accuracy was 9.95% at epoch 10 before the action and 10.46% at epoch 20 after the following generation (+0.51 percentage points); parameters increased from 6,200 to 8,536 after duplication. Seed 2 selected the duplicate at epoch 10, with candidate score 0.14; at its current 203-record snapshot it has 18.49% validation accuracy and 30,296 parameters. These short-window changes include continued training and later structural actions, so they do not isolate the duplicate's causal effect. No residual-block removal has been selected.

## Interim interpretation

- On MNIST, CIFAR-10, and CIFAR-100, the enabled arm's mean test accuracy is lower than the disabled arm. However, **none of those 18 completed runs applied a residual-block action**. They compare the effect of making the action available to search, not the effect of executing it.
- Candidate grades show that residual-block actions were evaluated, and duplicate candidates were chosen twice on Tiny ImageNet. This confirms the action can be selected in the larger dataset run, but there is only one completed enabled seed and no control yet.
- These results do not establish that residual-block actions help or hurt. The follow-up comparison is Experiment 013, limited to MNIST and CIFAR-10 with the Experiment 010 MCTS settings.

## Limitations and next step

- The report is interim: Tiny ImageNet enabled seed 2 is still running, and four Tiny ImageNet runs have not started.
- Three seeds provide an initial estimate; CIFAR-10 enabled results vary substantially by seed.
- The 90% early-stop target is suitable for MNIST but was not reached on CIFAR or completed Tiny ImageNet runs.
- Tiny ImageNet is a 200-class subset and does not test full ImageNet.
- The remaining Tiny ImageNet runs were not continued. Experiment 013 is the comparison to run next.
