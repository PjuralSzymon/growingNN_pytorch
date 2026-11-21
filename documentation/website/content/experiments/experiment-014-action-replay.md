---
title: "Experiment 14: replaying the epoch-80 decision"
---

All arms start from the Experiment 12 generation-3 pre-decision checkpoint, after epoch 80. Each recorded action and an unchanged control is trained for 20 epochs with the same nominal seed, training recipe and saved learning-rate schedule. The primary comparison uses 3,880 validation examples outside the original 1,120-example simulation validation subset.

The five actions and the original simulation scores are from the recorded decision. Resetting the seed does not guarantee identical random draws across different architectures. This is one checkpoint and one training seed; it is not a multi-seed estimate.

| Action | Original simulation | Parameters | Initial held-out | After 1 epoch | After 20 epochs | Difference vs control | Full validation after 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | — | 2,368 | 33.58% | 36.70% | 38.25% | +0.00 pp | 38.52% |
| Widen to 256 | 37.32% | 9,472 | 33.61% | 37.73% | 40.05% | +1.80 pp | 40.12% |
| Widen to 128 | 36.07% | 4,736 | 33.61% | 38.99% | 36.86% | -1.39 pp | 37.08% |
| Add linear | 34.91% | 6,528 | 33.58% | 31.93% | 41.91% | +3.66 pp | 41.64% |
| Conv without BN/ReLU | 25.98% | 39,296 | 33.58% | 28.48% | 39.90% | +1.65 pp | 39.70% |
| A08: conv + BN + ReLU | 39.15% | 39,424 | 33.58% | 45.00% | 56.80% | +18.56 pp | 56.80% |

![Held-out validation accuracy for all replay arms](/assets/experiments/014-action-replay-validation.png)

Full validation includes the original simulation-validation subset; held-out validation is the primary comparison. No test-set scores were used. The original run followed a different random trajectory, so its subsequent accuracy need not be reproduced exactly.


## Phase 2: scoring epochs

Each cell is validation accuracy after scoring a fresh copy of the same action, using the original simulation subset and the Monte Carlo rollout scoring path with rollout depth zero. The epoch settings are separate runs, using the same nominal seed; one score per action is not a stability estimate.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.43% | 39.20% | 32.50% | 29.55% | 34.82% |
| 5 | 39.29% | 38.66% | 39.38% | 29.91% | 48.30% |
| 10 | 37.14% | 38.21% | 41.96% | 36.70% | 41.43% |
| 20 | 37.23% | 40.00% | 40.00% | 36.79% | 48.57% |

![Simulation accuracy by scoring epochs and action](/assets/experiments/014-scoring-epochs.png)

These simulation scores use different data and training budgets than the full-training results above.


## Phase 3: Monte Carlo 2 search scores

Each row runs the complete Monte Carlo 2 search from the same checkpoint with the five recorded root actions. Cells show the final root-node mean search value and visit count; **bold** marks the selected action. These values combine rollout scores and are not necessarily direct validation accuracies. The search can include randomly chosen future actions. The configured 32-second budget is soft and may be exceeded while completing search.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3496 (n=1) | **0.3695 (n=1)** | 0.3293 (n=1) | 0.2540 (n=1) | 0.3429 (n=1) |
| 5 | 0.3777 (n=1) | 0.3776 (n=1) | 0.3316 (n=1) | 0.2621 (n=1) | **0.4311 (n=1)** |
| 10 | 0.3791 (n=1) | 0.3952 (n=1) | 0.3888 (n=1) | 0.3386 (n=1) | **0.4237 (n=1)** |
| 20 | 0.3996 (n=1) | 0.3986 (n=1) | 0.3713 (n=1) | 0.2504 (n=1) | **0.4835 (n=1)** |

![Monte Carlo 2 search values by scoring epochs](/assets/experiments/014-montecarlo2-epochs.png)

A run uses one nominal seed; differences may reflect search order, randomness and unequal visit counts as well as scoring epochs.

### What `MCTS_PROPAGATE_ROLLOUT_VALUE` means

**A value of `False` does not turn off rollouts.** Phase 3 inherited `False` from the project configuration; the Phase 3 script did not override it. The saved run also records `MCTS_ROLLOUT_DEPTH=2` and a depth-weight decay of `0.5`.

When an unvisited tree node is evaluated, `rollout()` scores its starting model, then scores after each randomly selected future action, up to the configured depth or until no actions are available. It returns their normalized weighted average, using weights `1`, `0.5`, and `0.25` when both future steps occur. This happens for either flag value. The saved results do not contain the individual actions or scores within each rollout.

The flag controls what an internal tree node returns to its parent **after** receiving and adding a child result:

- **`True`:** return the latest child result to the parent.
- **`False`:** return the internal node’s accumulated value to the parent. Once a node has multiple visits, this can propagate prior results again, so ancestor values may not be simple averages of individual rollout scores.

For example, if an internal node already has `0.4` and receives a new result of `0.6`, it adds the new result and holds `1.0`. With `True` it returns `0.6` upward; with `False` it returns `1.0`. This flag concerns propagation, not whether the `0.6` rollout ran.

In the saved Phase 3 runs, each root action was visited once and the reported maximum **tree depth** was one. Each root-child score therefore came from its own rollout; the cumulative-return issue would matter when propagation passes through revisited internal nodes in a deeper tree. Tree depth and the number of random actions inside a rollout are different quantities.


## Phase 4: Monte Carlo 2 time budgets

All rows use 20 scoring epochs. The baseline reuses the 20-epoch Phase 3 run. The other rows use nominal budgets of 20, 25 and 30 minutes; 5, 10 and 15 minutes were omitted because the baseline alone took approximately 17 minutes to complete its required initial evaluations. The budget is soft, so actual time can be longer.

Cells show the root search value and visit count; **bold** marks the selected action. Values can reflect random future actions and the algorithm’s value propagation, and are not direct validation accuracies. Twenty epochs gave the largest A08 margin in the single Phase 3 run; that does not establish it is the best general setting.

| Time budget | Actual time | Tree depth | Rollouts | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 seconds (Phase 3) | 17.0 min | 1 | 6 | 0.3996 (n=1) | 0.3986 (n=1) | 0.3713 (n=1) | 0.2504 (n=1) | **0.4835 (n=1)** |
| 20 minutes | 21.3 min | 2 | 8 | 0.4054 (n=1) | 0.4292 (n=2) | 0.3778 (n=1) | 0.3690 (n=1) | **0.4783 (n=2)** |
| 25 minutes | 26.7 min | 2 | 11 | 0.4009 (n=2) | 0.4010 (n=2) | 0.3679 (n=2) | 0.3181 (n=2) | **0.4777 (n=2)** |
| 30 minutes | 33.1 min | 2 | 13 | 0.3813 (n=3) | 0.3979 (n=2) | 0.3564 (n=2) | 0.3355 (n=2) | **0.4823 (n=3)** |

![Monte Carlo 2 search values by time budget](/assets/experiments/014-montecarlo2-time.png)

Each setting uses one nominal seed. Greater budgets may allow deeper exploration, but do not guarantee equal visits or identical random trajectories.


## Phase 5: full growth from the Experiment 010 starter

CIFAR-10, the Experiment 010 tiny starter (296 parameters), 10 generations of 20 training epochs, and seeds 1 and 2. This uses the Experiment 010 Pass 2 selected settings with Monte Carlo 2, **20 scoring epochs** and a **5-minute simulation budget**. The budget is soft and may be exceeded while the search completes its minimum evaluations.

Other settings: UCB C=1.0, legacy UCB formula, rollout propagation enabled, rollout depth 1, minimum iterations 3, and a 10% simulation training subset. Rollout epochs and rollout learning rate follow the Experiment 010 selected values but Monte Carlo 2 does not use them for intermediate training. Its depth-weight decay is the current configured value, since Experiment 010 did not tune that parameter.

| Seed | Status | Final validation | Final test | Best validation |
| ---: | --- | ---: | ---: | ---: |
| 1 | Complete | 64.02% | 63.92% | 64.86% |
| 2 | Pending | — | — | — |

Experiment 010 selected settings in sequential sweeps; its report does not provide a completed joint run of the final configuration as a matched baseline. The proposed 40-generation confirmation was not started. Differences in the algorithm and the two simulation-budget settings limit causal attribution.


### Seed 1: training history and action decisions

![Seed 1 training accuracy and model parameters](/assets/experiments/014-full-seed1-training.png)

The dotted lines mark generation boundaries. The change after 20 epochs includes the selected action **and continued training**; it is not an isolated causal effect.

| Generation ending | Before action validation | Parameters | Selected action | Next generation validation | Change after training | Candidates | Search time |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 0 (epoch 20) | 32.82% | 296 | A001: Sequential convolution + BN — conv1 → linear2 / 8→8, 3×3 | 44.02% | +11.20 pp | 3 | 7.3 min |
| 1 (epoch 40) | 44.02% | 896 | A004: Sequential convolution + BN — conv1 → seq_conv_0 / 8→8, 3×3 | 51.40% | +7.38 pp | 9 | 19.3 min |
| 2 (epoch 60) | 51.40% | 1,496 | A013: Sequential convolution + BN — seq_conv_1 → seq_conv_0 / 8→8, 3×3 | 49.92% | -1.48 pp | 17 | 34.6 min |
| 3 (epoch 80) | 49.92% | 2,096 | A028: Residual convolution — conv1 → seq_conv_2 / 8→8, 3×3 | 55.60% | +5.68 pp | 26 | 52.7 min |
| 4 (epoch 100) | 55.60% | 2,680 | A046: Residual convolution — conv1 → seq_conv_1 / 8→8, 3×3 | 58.86% | +3.26 pp | 39 | 61.7 min |
| 5 (epoch 120) | 58.86% | 3,264 | A065: Sequential convolution without BN — res_conv__1 → seq_conv_2 / 8→8, 3×3 | 59.80% | +0.94 pp | 53 | 103.0 min |
| 6 (epoch 140) | 59.80% | 3,848 | A072: Residual convolution — conv1 → seq_conv_0 / 8→8, 3×3 | 62.68% | +2.88 pp | 64 | 133.0 min |
| 7 (epoch 160) | 62.68% | 4,432 | A120: Sequential convolution without BN — conv1 → res_conv__2 / 8→8, 3×3 | 63.74% | +1.06 pp | 75 | 176.5 min |
| 8 (epoch 180) | 63.74% | 5,016 | A138: Residual convolution — conv1 → res_conv__2 / 8→8, 3×3 | 64.02% | +0.28 pp | 83 | 199.6 min |

### Seed 1: distribution of candidate scores

Each dot is one recorded candidate; the black-edged star is the chosen action. Family positions stay fixed in every generation. These are search scores, not the individual accuracy measurements inside each rollout. One visit can combine multiple model scores; the CSV includes the board’s separate latest rollout accuracy.

![Seed 1 candidate score distributions](/assets/experiments/014-full-seed1-scores-overview.png)

Individual action charts: [generation 0](/assets/experiments/014-full-seed1-scores-gen-0.png), [generation 1](/assets/experiments/014-full-seed1-scores-gen-1.png), [generation 2](/assets/experiments/014-full-seed1-scores-gen-2.png), [generation 3](/assets/experiments/014-full-seed1-scores-gen-3.png), [generation 4](/assets/experiments/014-full-seed1-scores-gen-4.png), [generation 5](/assets/experiments/014-full-seed1-scores-gen-5.png), [generation 6](/assets/experiments/014-full-seed1-scores-gen-6.png), [generation 7](/assets/experiments/014-full-seed1-scores-gen-7.png), [generation 8](/assets/experiments/014-full-seed1-scores-gen-8.png).

[Download all recorded candidate scores and full action descriptions](/assets/experiments/014-full-seed1-candidates.csv).

| Generation | Sequential conv with BN + ReLU | Sequential conv without BN/ReLU | Residual convolution | Sequential linear | Dropout | Delete layer | Total scored |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 0 | 1 | 0 | 0 | 3 |
| 1 | 2 | 2 | 1 | 1 | 3 | 0 | 9 |
| 2 | 3 | 3 | 3 | 1 | 6 | 1 | 17 |
| 3 | 4 | 4 | 6 | 1 | 9 | 2 | 26 |
| 4 | 6 | 6 | 8 | 1 | 15 | 3 | 39 |
| 5 | 8 | 8 | 11 | 1 | 21 | 4 | 53 |
| 6 | 9 | 9 | 16 | 1 | 24 | 5 | 64 |
| 7 | 11 | 11 | 17 | 2 | 27 | 7 | 75 |
| 8 | 12 | 12 | 19 | 2 | 30 | 8 | 83 |

Counts are scored candidates, not proof of how many were generated or available. Blank or zero means none of that type is recorded as scored.
