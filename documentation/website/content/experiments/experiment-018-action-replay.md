---
title: "Experiment 18: replaying the epoch-80 decision"
---

All arms start from the Experiment 17 generation-3 pre-decision checkpoint, after epoch 80. Each recorded action and an unchanged control is trained for 20 epochs with the same nominal seed, training recipe and saved learning-rate schedule. The primary comparison uses 3,880 validation examples outside the original 1,120-example simulation validation subset.

The five actions and the original simulation scores are from the recorded decision. Resetting the seed does not guarantee identical random draws across different architectures. This is one checkpoint and one training seed; it is not a multi-seed estimate.

| Action | Original simulation | Parameters | Initial held-out | After 1 epoch | After 20 epochs | Difference vs control | Full validation after 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | — | 2,368 | 33.58% | 36.70% | 38.25% | +0.00 pp | 38.52% |
| Widen to 256 | 37.32% | 9,472 | 33.61% | 37.73% | 40.05% | +1.80 pp | 40.12% |
| Widen to 128 | 36.07% | 4,736 | 33.61% | 38.99% | 36.86% | -1.39 pp | 37.08% |
| Add linear | 34.91% | 6,528 | 33.58% | 31.93% | 41.91% | +3.66 pp | 41.64% |
| Conv without BN/ReLU | 25.98% | 39,296 | 33.58% | 28.48% | 39.90% | +1.65 pp | 39.70% |
| A08: conv + BN + ReLU | 39.15% | 39,424 | 33.58% | 45.00% | 56.80% | +18.56 pp | 56.80% |

![Held-out validation accuracy for all replay arms](/assets/experiments/018-action-replay-validation.png)

Full validation includes the original simulation-validation subset; held-out validation is the primary comparison. No test-set scores were used. The original run followed a different random trajectory, so its subsequent accuracy need not be reproduced exactly.


## Phase 2: scoring epochs

Each cell is validation accuracy after scoring a fresh copy of the same action, using the original simulation subset and the Monte Carlo rollout scoring path with rollout depth zero. The epoch settings are separate runs, using the same nominal seed; one score per action is not a stability estimate.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.43% | 39.20% | 32.50% | 29.55% | 34.82% |
| 5 | 39.29% | 38.66% | 39.38% | 29.91% | 48.30% |
| 10 | 37.14% | 38.21% | 41.96% | 36.70% | 41.43% |
| 20 | 37.23% | 40.00% | 40.00% | 36.79% | 48.57% |

![Simulation accuracy by scoring epochs and action](/assets/experiments/018-scoring-epochs.png)

These simulation scores use different data and training budgets than the full-training results above.


## Phase 3: Monte Carlo 2 search scores

Each row runs the complete Monte Carlo 2 search from the same checkpoint with the five recorded root actions. Cells show the final root-node mean search value and visit count; **bold** marks the selected action. These values combine rollout scores and are not necessarily direct validation accuracies. The search can include randomly chosen future actions. The configured 32-second budget is soft and may be exceeded while completing search.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3496 (n=1) | **0.3695 (n=1)** | 0.3293 (n=1) | 0.2540 (n=1) | 0.3429 (n=1) |
| 5 | 0.3777 (n=1) | 0.3776 (n=1) | 0.3316 (n=1) | 0.2621 (n=1) | **0.4311 (n=1)** |
| 10 | 0.3791 (n=1) | 0.3952 (n=1) | 0.3888 (n=1) | 0.3386 (n=1) | **0.4237 (n=1)** |
| 20 | 0.3996 (n=1) | 0.3986 (n=1) | 0.3713 (n=1) | 0.2504 (n=1) | **0.4835 (n=1)** |

![Monte Carlo 2 search values by scoring epochs](/assets/experiments/018-montecarlo2-epochs.png)

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

![Monte Carlo 2 search values by time budget](/assets/experiments/018-montecarlo2-time.png)

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

![Seed 1 training accuracy and model parameters](/assets/experiments/018-full-seed1-training.png)

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

![Seed 1 candidate score distributions](/assets/experiments/018-full-seed1-scores-overview.png)

Individual action charts: [generation 0](/assets/experiments/018-full-seed1-scores-gen-0.png), [generation 1](/assets/experiments/018-full-seed1-scores-gen-1.png), [generation 2](/assets/experiments/018-full-seed1-scores-gen-2.png), [generation 3](/assets/experiments/018-full-seed1-scores-gen-3.png), [generation 4](/assets/experiments/018-full-seed1-scores-gen-4.png), [generation 5](/assets/experiments/018-full-seed1-scores-gen-5.png), [generation 6](/assets/experiments/018-full-seed1-scores-gen-6.png), [generation 7](/assets/experiments/018-full-seed1-scores-gen-7.png), [generation 8](/assets/experiments/018-full-seed1-scores-gen-8.png).

[Download all recorded candidate scores and full action descriptions](/assets/experiments/018-full-seed1-candidates.csv).

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


## Phase 6: replaying the A08 decision with the fixed Monte Carlo 2

Phase 5 did not reproduce an A08-like jump. Two causes were found: convolution widening was not enabled in Phase 5, so every added convolution stayed at 8 → 8 channels and no action comparable to A08 (a 64 → 64 convolution, +37k parameters) existed; and every candidate received exactly one visit, so the search reduced to "score each action once" with near-tied scores from generation 3 onwards.

The following issues were fixed in the Monte Carlo 2 simulation algorithm (`montecarlo_alg_2.py`):

- **True mean values.** Each tree node now passes the latest rollout value to its parent, so `value / visits` is a real mean. Previously, with `MCTS_PROPAGATE_ROLLOUT_VALUE=False`, nodes returned their accumulated sum and ancestors added sums of sums. Monte Carlo 2 now ignores that flag.
- **Mean-based UCB.** The legacy (non-square-root) UCB formula now exploits the mean value instead of the raw sum, matching the final choice, which uses the mean.
- **No root rollout.** The starting model is no longer rolled out before the search; its score was never used to choose an action. Covering N root actions now takes N rollouts instead of N + 1.
- **Score memory.** Only the node's initial score is remembered for reuse; scores of random rollout steps were stored under keys that could never be reused.
- **Board records.** Each candidate's recorded composite score is the weighted rollout value used by the search (not the last step's score), and the chosen row is marked by tree node rather than by its action text.

This phase repeats the Phase 3 search on the same Experiment 017 generation-3 checkpoint and five recorded root actions, with the fixed algorithm and the settings planned for Phase 7: **rollout depth 0**, **5 scoring epochs**, convolution widening enabled, UCB C=1.0, minimum iterations 3 and a 32-second soft budget. Three search seeds are run to show how stable the ranking is. Cells show the mean search value and visit count; **bold** marks the selected action.

| Seed | Search time | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.7 min | 0.3920 (n=1) | 0.3777 (n=1) | 0.3643 (n=1) | 0.2982 (n=1) | **0.4545 (n=1)** | A08: conv + BN + ReLU |
| 2 | 0.6 min | 0.3973 (n=1) | **0.4134 (n=1)** | 0.3196 (n=1) | 0.3625 (n=1) | 0.4134 (n=1) | Widen to 128 |
| 3 | 0.7 min | 0.3911 (n=1) | 0.4277 (n=1) | 0.3464 (n=1) | 0.3357 (n=1) | **0.4696 (n=1)** | A08: conv + BN + ReLU |

**Result:** A08 had the highest score, alone or tied, in 3 of 3 seeds. Exact ties occurred in seed 2 (Widen to 128, A08: conv + BN + ReLU); the search keeps the first tied candidate in action order, so a tie can select another action.

Each search scores every root action once (one rollout per action, no random follow-up actions), so one search costs one 5-epoch scoring run per candidate instead of two or three 20-epoch runs in Phase 5. Scores are single measurements on the 1,120-example simulation-validation subset and remain noisy below the top candidate.


## Phase 7: full growth with the fixed Monte Carlo 2

The Phase 5 recipe repeated with the fixed algorithm and the Phase 6 settings: convolution widening enabled, rollout depth 0 and 5 scoring epochs. All other settings match Phase 5, including the 5-minute soft search budget.

| Seed | Status | Final validation | Final test | Best validation |
| ---: | --- | ---: | ---: | ---: |
| 1 | Complete | 59.84% | 60.11% | 60.48% |
| 2 | Pending | — | — | — |

### Seed 1 compared with Phase 5

| Run | Final validation | Final test | Best validation | Final parameters | Total search time | Longest search |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 5 | 64.02% | 63.92% | 64.86% | 5,600 | 788 min | 199.6 min |
| Phase 7 | 59.84% | 60.11% | 60.48% | 3,392 | 50 min | 7.6 min |

![Phase 7 and Phase 5 seed 1 validation accuracy and parameters](/assets/experiments/018-phase7-seed1-vs-phase5.png)

### Seed 1 decisions

Mean score is the selected action's mean search value. Widening shows how many widening candidates passed the output-preservation check and the best rank one reached (by mean score). The change after 20 epochs includes continued training; it is not an isolated effect of the action.

| Generation ending | Validation before | Parameters | Selected action | Mean score | Candidates | Widening (best rank) | Rollouts | Tree depth | Search time | Change after training |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (epoch 20) | 33.32% | 296 | Sequential convolution + BN — conv1 → linear2 / 8→8, 3×3 | 0.3502 | 5 | 2 (#2) | 40 | 2 | 5.1 min | +12.14 pp |
| 1 (epoch 40) | 45.46% | 896 | Sequential convolution + BN — seq_conv_0 → linear2 / 8→8, 3×3 | 0.4415 | 10 | 1 (#5) | 39 | 2 | 5.1 min | +1.26 pp |
| 2 (epoch 60) | 46.72% | 1,496 | Residual convolution — conv1 → seq_conv_1 / 8→8, 3×3 | 0.5101 | 18 | 1 (#9) | 41 | 2 | 5.1 min | +6.48 pp |
| 3 (epoch 80) | 53.20% | 2,080 | Dropout — Dropout 0.2 / conv1 → res_conv__0 | 0.5406 | 28 | 0 | 41 | 2 | 5.2 min | -1.18 pp |
| 4 (epoch 100) | 52.02% | 2,080 | Residual convolution — conv1 → seq_conv_0 / 8→8, 3×3 | 0.5375 | 25 | 0 | 40 | 2 | 5.1 min | +2.84 pp |
| 5 (epoch 120) | 54.86% | 2,664 | Dropout — Dropout 0.1 / conv1 → res_conv__1 | 0.5634 | 38 | 0 | 40 | 2 | 5.4 min | +1.60 pp |
| 6 (epoch 140) | 56.46% | 2,664 | Sequential linear — res_conv__0 → linear2 / 8→8 | 0.5661 | 35 | 0 | 40 | 2 | 5.2 min | +1.36 pp |
| 7 (epoch 160) | 57.82% | 2,736 | Residual convolution — conv1 → seq_conv_1 / 8→8, 3×3 | 0.5857 | 47 | 0 | 47 | 1 | 6.2 min | +1.92 pp |
| 8 (epoch 180) | 59.74% | 3,320 | Sequential linear — seq_linear_0 → linear2 / 8→8 | 0.6009 | 58 | 0 | 58 | 1 | 7.6 min | +0.10 pp |

The board files of this run mark the wrong candidate as `chosen` (a reporting bug in the fixed search, corrected afterwards: the board sorts candidate rows, and rows were matched to tree nodes by position). The executed action was always the highest mean score; this table uses the recorded executed action, not the flag.

### What seed 1 shows

- **Search is about 16 times cheaper in total.** Each search took 5.1–7.6 minutes, against up to 200 minutes in Phase 5, because each candidate costs one 5-epoch scoring run.
- **Spare budget was used to search deeper.** In generations 0, 1, 2, 3, 4, 5 and 6 the search covered the root actions and then expanded the best ones (tree depth 2). In generations 7 and 8 the 47–58 candidates used the whole budget at one visit each.
- **Widening was offered in generations 0, 1 and 2 (best rank 2–9) and never selected.** From generation 3 onward no widening passed the output-preservation check: the widening code used by this run only handled a single path, and a residual convolution after `conv1` gives it two consumers. Fan-out widening, which handles that case, was added to `WidenConv` after this run.
- **Close scores still decide.** The top two mean scores differed by less than 1 pp in 8 of 9 generations, within the noise of the 1,120-example simulation-validation subset. Weaker actions were selected: dropout in generations 3 and 5; sequential linear in generations 6 and 8. The model ended with 3,392 parameters against 5,600 in Phase 5, and its final validation accuracy was 4.18 pp lower.


Run with `--phase full-score-fix`, then regenerate this report.


## Phase 8: replaying the A08 decision with 20 scoring epochs

Phase 6 repeated with **20 scoring epochs** instead of 5; everything else is unchanged (fixed Monte Carlo 2, rollout depth 0, convolution widening, UCB C=1.0, minimum iterations 3, 32-second soft budget, seeds 1–3). Phase 6 showed that one 5-epoch score per action varies by several points between seeds, so A08 tied with Widen 128 in seed 2. This phase measures whether longer scoring makes the scores more stable and A08's lead clearer.

| Seed | Search time | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3.5 min | 0.3741 (n=1) | 0.4179 (n=1) | 0.3768 (n=1) | 0.3714 (n=1) | **0.4679 (n=1)** | A08: conv + BN + ReLU |
| 2 | 4.3 min | 0.4232 (n=1) | **0.4295 (n=1)** | 0.4161 (n=1) | 0.3429 (n=1) | 0.4286 (n=1) | Widen to 128 |
| 3 | 4.5 min | 0.3705 (n=1) | 0.4054 (n=1) | 0.3777 (n=1) | 0.3464 (n=1) | **0.4500 (n=1)** | A08: conv + BN + ReLU |

**Result:** A08 had the highest score, alone or tied, in 2 of 3 seeds.

### 5 vs 20 scoring epochs

Mean score across seeds, with the lowest and highest seed in brackets. A narrower range means a single scoring run is more reliable.

| Action | 5 epochs (Phase 6) | 20 epochs (Phase 8) |
| --- | ---: | ---: |
| Widen to 256 | 39.35% (39.11–39.73) | 38.93% (37.05–42.32) |
| Widen to 128 | 40.62% (37.77–42.77) | 41.76% (40.54–42.95) |
| Add linear | 34.35% (31.96–36.43) | 39.02% (37.68–41.61) |
| Conv without BN/ReLU | 33.21% (29.82–36.25) | 35.36% (34.29–37.14) |
| A08: conv + BN + ReLU | 44.58% (41.34–46.96) | 44.88% (42.86–46.79) |
| **A08 lead over the next-best action, per seed** | +6.2, +0.0, +4.2 pp | +5.0, -0.1, +4.5 pp |
| Search time per seed | 0.7 min | 4.1 min |

### What Phase 8 shows

- **20 scoring epochs did not make a single score reliable.** A08 again lost seed 2, now by about one validation image, and the spread of each action across seeds is about as wide as with 5 epochs. Seed 2 was unfavourable for A08 in both settings, so the noise comes mostly from the random training draw of one scoring run, not from too few epochs.
- **Averaging seeds does separate A08.** Averaged over the three seeds, A08 leads the next-best action by +4.0 pp with 5 epochs and +3.1 pp with 20 epochs.
- **Averaging is also cheaper.** Three 5-epoch runs per action take about 2.0 minutes for these five actions, less than one 20-epoch run (4.1 minutes).
- **Implication for the search:** re-scoring the top few candidates with different seeds and averaging is a better use of compute than longer scoring. This was not yet tested in a full-growth run.


## Phase 9: replaying the A08 decision with noise fixes and a 5-minute budget

Phases 6 and 8 stopped as soon as each root action had one rollout, so they measured a single noisy score per action and no future actions. Phase 9 uses the same five root actions, 5 scoring epochs, rollout depth 0 and seeds 1–3, with four changes:

- **5-minute budget** (instead of 32 seconds), so the search keeps going after the first pass: it expands root actions and scores "action + next action" models, which both measures the future and adds more measurements per root action.
- **Type-balanced expansion order.** New children are tried in a random order that cycles through action types, instead of the fixed action-list order.
- **Type-balanced random follow-ups** (inactive here, since rollout depth is 0). Rollout follow-ups pick an action type first, so dropout variants no longer dominate.
- **One training seed per search, and a 2-epoch score.** Every scoring run in a search uses the same training seed, so differences come from the architectures; the score averages the last two scoring epochs.

Cells show the mean search value and visit count; **bold** marks the selected action.

| Seed | Search time | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 5.1 min | 0.3592 (n=6) | 0.4023 (n=6) | 0.3772 (n=6) | 0.3079 (n=5) | **0.4184 (n=6)** | A08: conv + BN + ReLU |
| 2 | 5.2 min | 0.3837 (n=5) | 0.3724 (n=5) | 0.3694 (n=5) | 0.2957 (n=5) | **0.4470 (n=6)** | A08: conv + BN + ReLU |
| 3 | 5.1 min | 0.3720 (n=6) | 0.3834 (n=6) | 0.3240 (n=6) | 0.2983 (n=6) | **0.4142 (n=6)** | A08: conv + BN + ReLU |

**Result:** A08 had the highest score, alone or tied, in 3 of 3 seeds.

| Seed | Rollouts | Tree depth | A08 lead over the next-best action |
| ---: | ---: | ---: | ---: |
| 1 | 29 | 2 | +1.6 pp |
| 2 | 26 | 2 | +6.3 pp |
| 3 | 30 | 2 | +3.1 pp |

For comparison, Phase 6 (one rollout per action) had A08 leads of +6.2, +0.0, +4.2 pp.

### What Phase 9 shows

- **A08 was selected in 3 of 3 seeds**, with a strict lead in each. This is the first multi-seed replay in which the search picked A08 every time.
- **The weakest seed improved, the average lead did not change much.** The smallest A08 lead rose from +0.0 pp (Phase 6) to +1.6 pp; the mean lead was +3.5 pp in Phase 6 and +3.7 pp here. The gain is reliability, not a bigger gap.
- **Every action got about the same number of visits** (5–6 each, 26–30 rollouts per search). With UCB C=1.0 the exploration bonus is roughly ten times larger than the score gaps of 0.02–0.06, so UCB spread the visits evenly. In effect each action was measured about six times, itself plus "action + next action" models, and the averages were compared.
- **Scores are lower than in Phase 6 by design.** Each average now includes follow-up models and the mean of the last two scoring epochs, so only the ranking is comparable with earlier phases, not the absolute values.
- **The four changes were tested together**, so this phase does not show which one mattered. The longer budget is the most likely main factor, because it is what gives each action several measurements.
- **Caveat for full growth.** One rollout took 10–12 seconds here, so a 5-minute search manages about 30 rollouts. This replay has only five root actions; full-growth generations had 18–58, so most would still get a single visit within five minutes. Getting the same benefit there needs a budget that grows with the number of candidates, or a smaller UCB C so that visits concentrate on the leaders.


## Phase 10: UCB C and simulation-set grid (full growth)

Full growth from the Experiment 010 starter, as in Phase 7, with the Phase 9 search: fixed Monte Carlo 2, type-balanced expansion, one training seed per search, the mean of the last two of 5 scoring epochs, rollout depth 0, and convolution widening including fan-out widening. The search budget grows with the number of candidates: 24 seconds per candidate (about two rollouts), at least 5 minutes.

Two factors are crossed:

- **UCB C = 0.05 or 1.0.** Score gaps between good actions are 0.01–0.05. With C = 1.0 the exploration bonus is about ten times larger, so visits spread evenly (Phase 9). With C = 0.05, extra visits concentrate on the leading candidates.
- **Simulation set = Moderate or CCS** (Experiment 011). Moderate was best and most stable on CIFAR-10 (58.4%, seeds 57.9–58.8%); CCS was best over CIFAR-10 and MNIST together (75.2%), mostly from MNIST, and varied strongly between CIFAR-10 seeds (46.1–61.4%).

Seed 1 is run for all four cells first. Each factor is judged on two runs against two; seed 2 repeats the grid if time allows.

| UCB C | Simulation set | Seed 1 validation | Seed 2 validation | Mean validation | Mean test | Final parameters | Search time per run |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | Moderate | 60.76% | Pending | 60.76% | 60.08% | 5,000 | 141 min |
| 0.05 | CCS | 59.38% | Pending | 59.38% | 59.62% | 3,904 | 128 min |
| 1.0 | Moderate | 61.92% | Pending | 61.92% | 62.15% | 5,752 | 101 min |
| 1.0 | CCS | 63.06% | Pending | 63.06% | 63.18% | 5,032 | 157 min |

Average final validation over the complete seeds (1):

| Factor | Level | Mean validation |
| --- | --- | ---: |
| UCB C | 0.05 | 60.07% |
| UCB C | 1.0 | 62.49% |
| Simulation set | Moderate | 61.34% |
| Simulation set | CCS | 61.22% |

### Training histories

Solid lines are validation accuracy, dashed lines training accuracy. The best cell (C=1.0, CCS) is compared with the other UCB C at the same simulation set, and with the other simulation set at the same C.

![UCB C compared](/assets/experiments/018-grid-training-ucb.png)

![Simulation set compared](/assets/experiments/018-grid-training-simset.png)

### Factor means

![Phase 10 mean final validation per factor level](/assets/experiments/018-grid-factor-means.png)

Each bar averages the two runs at that level (one per level of the other factor); dots are the individual runs.

### Could the search tell its candidates apart?

![How clearly each search separated its candidates](/assets/experiments/018-grid-separation.png)

Each dot is one candidate, placed by how far its score was below the best candidate of that search. The grey band is the 1 pp noise level of one scoring run; the number above each generation counts the candidates inside it. A search that knows what is best has one dot at 0 and the rest well below; many dots inside the band mean the choice was effectively a coin toss among near-ties.

### What the selected actions delivered

![What each selected action delivered](/assets/experiments/018-grid-action-gains.png)

Validation change over the generation after each decision, for every decision in the four cells, by kind of action. The grey line is Experiment 17 for contrast: its generation-3 action (A08, 64 → 64 convolution after widening to 64 channels) gave the +20.8 pp jump.

### Candidate scores in the best cell

![Candidate scores, C=1.0, CCS](/assets/experiments/018-grid-c1p0_ccs-scores.png)

Each row is one generation and each dot one candidate, grouped by action type; the black-edged star is the selected action.

### Seed 1 decisions

Selected action type in each generation, with the rank of the best widening candidate (by mean score) in brackets.

| Generation | C=0.05, Moderate | C=0.05, CCS | C=1.0, Moderate | C=1.0, CCS |
| ---: | --- | --- | --- | --- |
| 0 | Sequential convolution + BN (#3) | Sequential convolution + BN (#3) | Sequential convolution + BN (#2) | Sequential convolution + BN (#2) |
| 1 | Sequential convolution + BN (#6) | Sequential convolution without BN (#2) | Sequential convolution + BN (#4) | Sequential convolution + BN (#7) |
| 2 | Residual convolution (#8) | Sequential convolution + BN (#3) | Sequential linear (#7) | Residual convolution (#6) |
| 3 | Residual convolution (#6) | Residual convolution (#6) | Add neurons (#19) | Sequential convolution + BN (#5) |
| 4 | Dropout (#13) | Dropout (#2) | Sequential convolution + BN (#8) | Sequential convolution + BN (#4) |
| 5 | Residual convolution (#11) | Residual convolution (#4) | Dropout (#10) | Residual convolution (#12) |
| 6 | Residual convolution (#27) | Dropout (#5) | Add neurons (#6) | Residual convolution (#17) |
| 7 | Sequential convolution without BN (#8) | Residual convolution (#11) | Widen convolution (#1) | Dropout (#18) |
| 8 | Residual convolution (#5) | Sequential linear (#15) | Residual convolution (#–) | Sequential convolution without BN (#8) |

### What Phase 10 shows

- **Final validation ranged from 59.38% to 63.06%.** No cell beat Phase 5 (64.02%) and all stayed far below Experiment 17 (69.48%). C = 1.0 ended higher than C = 0.05 with both simulation sets; with one seed per cell this is within the seed-to-seed spread. Moderate and CCS were practically equal.
- **Nothing useful was added after generation 2.** Generations 0–2 gained +6.9 pp on average. From generation 3, generations where the selected action added a layer gained +1.4 pp, and generations where it added no capacity (dropout, neuron or layer removal) gained +1.9 pp. The added layers brought nothing beyond continued training.
- **The models are capacity-limited, not overfitting.** Final training accuracy was 58.2–62.0% against 59.4–63.1% validation (training is measured with cutout augmentation). The networks cannot fit even the training data, so only a real capacity increase can help.
- **The network stayed 8 channels wide.** The only widening chosen took conv1 to 16 channels in generation 7 of C = 1.0, Moderate, too late to matter. Every added convolution is 8 → 8 (about +600 parameters) and everything reaches the classifier through an 8-channel global pool. Final models had 3,904–5,752 parameters. Experiment 17 widened conv1 to 64 channels in generations 0–2 and had 39,424 parameters after its generation-3 action.
- **The score cannot see the value of widening.** Widening was offered in 35 of 36 searches and chosen 1 time. Its best score was on average -0.3 pp from a nearly-no-change action (dropout 0.1), while the best added layer scored +1.6 pp above it. Widening copies channels so the function is unchanged, and the copies only drift apart slowly, so within 5 scoring epochs a widened model behaves like the unwidened one. An added layer gets a small short-term edge that does not last.
- **The scores flatten as the model trains.** Every candidate starts from the same trained weights and changes it only slightly, so in generations 7–8 between 7 and 34 of 31–77 candidates scored within 1 pp of the best (the separation chart).
- **Late changes have little learning rate left.** The cosine schedule falls from 0.1 to 0.036 at the end of generation 5 and 0.0034 at the end of generation 8, so a new layer added late cannot learn much in the remaining epochs. Experiment 17 made its decisive change at generation 3, with the learning rate still at 0.066.
- **The budget bought fewer rollouts than planned:** 1.2 per candidate from generation 3, at about 20 s per rollout instead of 10–12 s in Phase 9. Expanding a node builds every child up front, which is the most likely cause (not profiled). As a result UCB C changed little after the first generations.

### What to change

The search is not failing to find a good action among those it compares; it compares the wrong things with a score that cannot see width. In order of expected impact:

1. **Make width growth reachable.** Offer widening by larger factors (up to 64 channels) and favour capacity when scores are within the 1 pp noise level, for example by picking the near-tied candidate that adds the most parameters. This targets both the 8-channel bottleneck and the near-ties.
2. **Score the gain over the unchanged model.** Score the current model once per search and rank candidates by improvement over it; if no candidate beats it by more than the noise, prefer the one that adds most capacity instead of a random near-tie.
3. **Decide big changes early or keep the learning rate up.** Either limit growth to the first half of the run, or restart the learning rate after each structural change, so late additions can still learn.
4. **Build children lazily** so the budget buys the intended two or more rollouts per candidate.

A cheap way to confirm the diagnosis before changing the search: one run that widens conv1 by 2× in each of generations 0–2 (as Experiment 17 did) and then lets the Phase 10 search decide. If it reaches Experiment 17 territory, width is the missing ingredient and items 1–2 are the right fix.


## Simulation fixes after Phase 10

The Phase 10 analysis traced the near-flat candidate scores to three problems in how Monte Carlo 2 built and trained candidates. All three are fixed in the code; Phase 11 checks them on the A08 decision. A fourth fix, measuring again before expanding, was added after Phase 11; Phase 12 runs full growth with all four.

| Fix | Problem found in Phase 10 | Change | Commit |
| --- | --- | --- | --- |
| **Lazy expansion** | Expanding a node built every child (about 70 model copies with every action applied) although the search scores one per step. Rollouts took about 20 s instead of 10–12 s, so the budget bought about 1.2 measurements per candidate. | A child model is built only when the search first visits it. The search rules and the order of children are unchanged. | `f8bed63` |
| **Scoring learning rate** | Scoring trained each candidate with the main run's learning rate at its current point: 0.098 in generation 0 but 0.010 and 0.0032 in generations 7–8. Late candidates barely changed during scoring, so up to 34 of 77 scored within 1 pp of the best. | Every scoring run trains with a restarted copy of the main schedule (`lr_scheduler.restarted()`), so candidates in every generation are trained the way the main run starts. The main run's schedule is never modified. | `de231f4` |
| **Futures continue from trained models** | Scoring trained a throwaway copy, so a child "X then Y" was built from the untrained parent and evaluated as X and Y applied at once, never as X → train → Y → train. An action that pays off one step later, such as widening, could not show its value. | `SimulationScore.score_and_train` also returns the copy its first training term trained; each node keeps it, and its children and rollout follow-ups are built from it. | `5d3c3f0` |
| **Measure again before expanding** *(added after Phase 11)* | Each node's own score was measured once, and the best of 40–77 single noisy scores tends to be a lucky one rather than the best action. | A node is measured `MCTS_MEASUREMENTS_BEFORE_EXPAND` times (default 2), measurement *i* with training seed *search seed + i*, before it is expanded; its value is the average. A re-measurement replaces an expansion visit, so a search makes the same number of scoring runs. | `abe0a33` |

Not adopted from the Phase 10 suggestions: rules that prefer adding capacity when scores are close (suggestions 1–2 above). The search must pick by its score alone, so that the same algorithm still works when the score rewards fewer parameters. The learning-rate schedule of the main run (suggestion 3) is a training-recipe question and is left for a separate experiment.

Expected effect, to be checked in the next run: late-generation candidates should separate instead of tying within 1 pp, and each search should measure more candidates more than once. One side effect to watch: a depth-2 candidate has had 10 scoring epochs in total against 5 for a root action, so deeper measurements may score higher.


## Phase 11: the A08 decision with the first three simulation fixes

The Phase 9 replay repeated with the code at `5d3c3f0`: lazy expansion, the restarted scoring learning rate and futures that continue from trained models. Everything else matches Phase 9: the Experiment 17 generation-3 checkpoint, the five recorded root actions, 5 scoring epochs with the mean of the last two, rollout depth 0, UCB C = 1.0, a 5-minute budget and seeds 1–3. This was a quick check run with a temporary script (saved with the results as `check_script.py`), not an experiment phase; measuring again before expanding (`abe0a33`) was not yet in the code.

| Seed | Selected | A08 score | Next best | A08 lead | Phase 9 lead | Phase 6 lead | Rollouts | Tree depth |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | A08: conv + BN + ReLU | 0.4508 | Widen to 256 (0.3792) | **+7.2 pp** | +1.6 pp | +6.2 pp | 24 | 3 |
| 2 | A08: conv + BN + ReLU | 0.4500 | Widen to 128 (0.4044) | **+4.6 pp** | +6.3 pp | +0.0 pp | 19 | 2 |
| 3 | A08: conv + BN + ReLU | 0.4287 | Widen to 256 (0.3629) | **+6.6 pp** | +3.1 pp | +4.2 pp | 22 | 3 |

Each action's mean search value, own score and best explored future, averaged over the three seeds:

| Action | Mean search value | Own score | Best explored future |
| --- | ---: | ---: | ---: |
| A08: conv + BN + ReLU | 0.4432 | 0.4388 | 0.4751 |
| Widen to 128 | 0.3695 | 0.3751 | 0.4259 |
| Widen to 256 | 0.3685 | 0.3744 | 0.4414 |
| Add linear | 0.3413 | 0.3494 | 0.3738 |
| Conv without BN/ReLU | 0.2791 | 0.2896 | 0.3378 |

### What Phase 11 shows

- **The gap between the good action and the rest is now bigger.** A08 was selected in all three seeds, and its lead over the next-best action grew from +3.7 pp on average in Phase 9 (smallest +1.6 pp) to **+6.1 pp** (smallest +4.6 pp). Phase 6 had +3.5 pp with a tie in one seed.
- **Futures now show what widening makes possible.** The two widening actions scored 0.375 on their own, but their best follow-up (mostly a convolution with BatchNorm and ReLU on the widened layer) reached 0.434. Before the fix this follow-up was built from untrained weights and could not show it.
- **Searches made fewer rollouts** in the same 5 minutes (19–24 against 26–30 in Phase 9; the cause was not measured) and reached tree depth 2–3.
- **What this does not show:** this decision is at generation 3, where the old scoring learning rate was still 0.066, so the learning-rate fix matters little here. Its effect, and the effect of measuring again, appear in late generations of a full-growth run. Deeper futures also have more scoring training in total (10 epochs at depth 2 against 5 at the root), so part of the higher future scores is extra training.


## Phase 12: full growth with the simulation fixes

The best Phase 10 cell (UCB C = 1.0, CCS simulation set) repeated with all four simulation fixes made after Phase 10: lazy expansion, the restarted scoring learning rate, futures that continue from trained models, and measuring each node twice before expanding it. Everything else matches Phase 10: the Experiment 010 starter, 10 generations of 20 epochs, 5 scoring epochs with the mean of the last two, rollout depth 0, convolution widening including fan-out, and 24 seconds of search per candidate (at least 5 minutes). Seeds 1 and 2.

| Run | Seed | Final validation | Final test | Final parameters | Search time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Phase 12** | 1 | **60.90%** | 61.04% | 4,448 | 137 min |
| **Phase 12** | 2 | **63.28%** | 64.19% | 5,616 | 156 min |
| Phase 10, C = 1.0, CCS | 1 | 63.06% | 63.18% | 5,032 | 157 min |
| Phase 5 (no widening, 20 scoring epochs) | 1 | 64.02% | 63.92% | 5,600 | 788 min |
| Experiment 17 (family search) | 1 | 69.48% | 69.83% | 80,887 | — |

Experiment 17's search time is not shown: its simulation files record only part of the family search.

![Phase 12 against the best Phase 10 cell](/assets/experiments/018-sim-fixes-training.png)

Solid lines are validation accuracy, dashed lines training accuracy.

### Did the candidates separate?

Number of candidates within 1 pp of the best candidate in each search (fewer means the search could tell the leader apart).

| Generation | Phase 10, C = 1.0, CCS | Phase 12 seed 1 | Phase 12 seed 2 |
| ---: | ---: | ---: | ---: |
| 0 | 1 of 5 | 1 of 5 | 2 of 5 |
| 1 | 1 of 10 | 2 of 10 | 1 of 10 |
| 2 | 3 of 18 | 4 of 18 | 3 of 18 |
| 3 | 5 of 31 | 5 of 29 | 3 of 27 |
| 4 | 1 of 41 | 6 of 40 | 1 of 37 |
| 5 | 3 of 52 | 1 of 50 | 1 of 49 |
| 6 | 4 of 65 | 1 of 61 | 3 of 60 |
| 7 | 34 of 77 | 6 of 58 | 3 of 74 |
| 8 | 25 of 74 | 5 of 55 | 7 of 90 |

### Best candidate score against the model being grown

Score of the selected candidate, and the validation accuracy of the model when the search ran (in brackets).

| Generation | Phase 10, C = 1.0, CCS | Phase 12 seed 1 | Phase 12 seed 2 |
| ---: | ---: | ---: | ---: |
| 0 | 37.1% (30.1%) | 36.6% (29.7%) | 32.2% (33.8%) |
| 1 | 43.4% (42.2%) | 41.9% (43.8%) | 45.7% (45.7%) |
| 2 | 47.1% (51.6%) | 48.2% (47.0%) | 48.8% (50.5%) |
| 3 | 51.3% (54.0%) | 49.1% (48.8%) | 51.7% (53.2%) |
| 4 | 53.2% (55.2%) | 52.7% (55.7%) | 53.2% (51.8%) |
| 5 | 55.9% (54.8%) | 52.6% (56.2%) | 54.3% (56.7%) |
| 6 | 58.7% (58.1%) | 54.6% (60.4%) | 54.3% (59.1%) |
| 7 | 59.0% (60.3%) | 52.3% (60.4%) | 53.3% (62.8%) |
| 8 | 59.3% (62.0%) | 52.0% (60.3%) | 53.2% (62.9%) |

### Decisions

Selected action type per generation, with the change in validation accuracy over the next generation.

| Generation | Phase 10, C = 1.0, CCS | Phase 12 seed 1 | Phase 12 seed 2 |
| ---: | --- | --- | --- |
| 0 | Sequential convolution + BN (+12.1 pp) | Sequential convolution + BN (+14.1 pp) | Sequential convolution + BN (+11.9 pp) |
| 1 | Sequential convolution + BN (+9.3 pp) | Sequential convolution + BN (+3.2 pp) | Sequential convolution + BN (+4.8 pp) |
| 2 | Residual convolution (+2.5 pp) | Residual convolution (+1.7 pp) | Sequential convolution + BN (+2.7 pp) |
| 3 | Sequential convolution + BN (+1.2 pp) | Sequential convolution + BN (+6.9 pp) | Sequential convolution + BN (-1.4 pp) |
| 4 | Sequential convolution + BN (-0.5 pp) | Sequential convolution + BN (+0.5 pp) | Sequential convolution without BN (+4.8 pp) |
| 5 | Residual convolution (+3.3 pp) | Residual convolution (+4.3 pp) | Residual convolution (+2.4 pp) |
| 6 | Residual convolution (+2.2 pp) | Dropout (-0.0 pp) | Residual convolution (+3.6 pp) |
| 7 | Dropout (+1.7 pp) | Dropout (-0.1 pp) | Residual convolution (+0.1 pp) |
| 8 | Sequential convolution without BN (+1.0 pp) | Residual convolution (+0.6 pp) | Residual convolution (+0.4 pp) |

### What Phase 12 shows

- **Growth slows after generation 3 in every run.** From the end of generation 3 to the end, validation accuracy rose by 9.0 pp (Phase 10, C = 1.0, CCS), 12.1 pp (Phase 12 seed 1), 10.1 pp (Phase 12 seed 2) over six generations. In generations 5–8 no candidate except widening added more than 600 parameters: every new convolution is 8→8 channels, because new layers take the channel count of the layers they connect, and widening doubles a single layer from 8 to 16 channels. Training accuracy stays at validation accuracy (seed 2 ended at 62.3% training, 63.3% validation), so the network is too small rather than over-fitting. The Phase 1 action that gave +18.6 pp was a 64-channel convolution; no action of that size is on offer here, so even a perfect search could only pick between small steps.
- **Final accuracy did not improve.** The two seeds ended at 60.90% and 63.28% validation (mean 62.09%), against 63.06% for the same cell in Phase 10. The two seeds differ by more than Phase 12 differs from Phase 10, and the four Phase 10 cells (one seed each) ranged from 59.38% to 63.06%, so the runs show no change in either direction.
- **Late-generation candidates now separate.** In generations 5–8, Phase 10 had on average 16.5 candidates within 1 pp of the best (up to 34 of 77 in generation 7); Phase 12 had 3.2 and 3.5. This is what the fixes were meant to do, but the better separation did not show up in the final accuracy.
- **Scores now sit below the model being grown.** With the restarted learning rate, scoring trains each candidate for 5 epochs at about 0.1 while the main training in generations 5–8 runs at about 0.035 down to 0.001. In generations 6–8 the selected candidate scored on average 7.4 and 8.0 pp below the model's validation accuracy, against 1.1 pp in Phase 10. The score now measures how a candidate does after a short high-learning-rate restart, and the main training then continues the low-learning-rate schedule instead. Whether the two rank candidates differently was not measured.
- **Seed 1 spent its late generations on Dropout.** It selected Dropout in generations 6 and 7, which added no parameters and left validation accuracy flat (60.4% and 60.3% after them). Dropout leading under high-learning-rate scoring fits the point above, but the cause was not measured.
- **Widening was never selected** (0 of 18 decisions), although it was runner-up in several searches; the runs ended at 4,400–5,600 parameters, far below Experiment 17's 80,887.
- **Searches went shallower.** Measuring each node twice uses rollouts that used to expand the tree: Phase 12 searches reached depth 1 from generation 1 on, Phase 10 reached depth 2. Search time was unchanged (137 and 156 min against 157 min).

### What to change

- **Offer larger growth steps.** Let new convolutions choose a wider output (for example 16, 32 or 64 channels, as in the Phase 1 action A08), or widen a whole chain of layers at once instead of one layer. This is a change to the actions, not to the simulation.
- **Keep the main learning-rate schedule.** Experiment 008 showed that one global schedule beats restarting at 0.1 every generation. The scoring restart stays a simulation-only choice; whether it ranks candidates the same way as the main schedule would is still open.
- **Run more seeds before comparing cells.** The two Phase 12 seeds differ by 2.4 pp; the four Phase 10 cells, one seed each, spread over 3.7 pp.


## Phase 13: every growing candidate of one late decision, trained for real

Why do generations after the third bring so little? Phase 13 takes the Phase 12 seed-2 search after generation 3 and trains every candidate that grows the network (all except Dropout, Delete layer and Delete neurons: 16 of 27) for a full 20-epoch generation on the full training set with the main learning-rate schedule, plus the unchanged model as a control. This shows how far the selected action was from the best one available, and whether any available action could have given a large gain.

Phase 12 saved no checkpoints, so the generation-3 model is grown again from the starter with the run's recorded choices and training recipe. It has the same architecture but not the same weights, because the original searches consumed random numbers between generations.

The rebuilt model reached 53.08% validation at the end of generation 3 (Phase 12 seed 2: 53.20%). The control trained one more generation without an action and reached 54.68%.

| Real rank | Action | Real validation | Gain over control | Simulation score | Simulation rank |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Sequential convolution + BN — seq_conv_1 → seq_conv_2 / 8→8, 3×3 | 57.76% | +3.08 pp | 0.4500 | 11 |
| 2 | Residual convolution — conv1 → seq_conv_0 / 8→8, 3×3 | 57.00% | +2.32 pp | 0.4714 | 8 |
| 3 | Sequential convolution + BN — seq_conv_0 → seq_conv_1 / 8→8, 3×3 | 56.44% | +1.76 pp | 0.4469 | 12 |
| 4 | Sequential convolution + BN — seq_conv_2 → linear2 / 8→8, 3×3 | 56.26% | +1.58 pp | 0.5004 | 4 |
| 5 | Residual convolution — seq_conv_0 → seq_conv_1 / 8→8, 3×3 | 56.10% | +1.42 pp | 0.3629 | 16 |
| 6 | Residual convolution — conv1 → seq_conv_1 / 8→8, 3×3 | 55.86% | +1.18 pp | 0.4797 | 7 |
| 7 | Widen convolution — conv1 → 16 ch | 55.80% | +1.12 pp | 0.4627 | 10 |
| 8 | Sequential convolution without BN — conv1 → seq_conv_0 / 8→8, 3×3 | 54.72% | +0.04 pp | 0.4699 | 9 |
| 9 | Sequential convolution + BN — conv1 → seq_conv_0 / 8→8, 3×3 **(selected)** | 54.30% | -0.38 pp | 0.5165 | 1 |
| 10 | Sequential convolution without BN — seq_conv_1 → seq_conv_2 / 8→8, 3×3 | 54.28% | -0.40 pp | 0.4103 | 15 |
| 11 | Residual convolution — conv1 → seq_conv_2 / 8→8, 3×3 | 52.80% | -1.88 pp | 0.5094 | 3 |
| 12 | Sequential convolution without BN — seq_conv_0 → seq_conv_1 / 8→8, 3×3 | 52.50% | -2.18 pp | 0.5096 | 2 |
| 13 | Residual convolution — seq_conv_1 → seq_conv_2 / 8→8, 3×3 | 51.26% | -3.42 pp | 0.4402 | 13 |
| 14 | Sequential convolution without BN — seq_conv_2 → linear2 / 8→8, 3×3 | 51.00% | -3.68 pp | 0.4839 | 5 |
| 15 | Residual convolution — seq_conv_0 → seq_conv_2 / 8→8, 3×3 | 50.08% | -4.60 pp | 0.4379 | 14 |
| 16 | Sequential linear — seq_conv_2 → linear2 / 8→8 | 48.38% | -6.30 pp | 0.4801 | 6 |

![Phase 13 training histories](/assets/experiments/018-decision-replay-training.png)

Every line starts from the same model and training seed and differs only in the action applied before epoch 1. The dashed black line is the control without an action. The middle panel subtracts the control, so a line above zero is ahead of it at that epoch. One candidate holds more than one role.


### What Phase 13 shows

- **The selected action ranked 9 of 16 in real training**, 3.46 pp below the best candidate (Sequential convolution + BN — seq_conv_1 → seq_conv_2 / 8→8, 3×3).
- **The best available gain over the control was +3.08 pp**; the middle candidate gained +0.04 pp and the worst -6.30 pp.
- **Simulation and real ranking agree with a rank correlation of -0.15** (1 is the same order, 0 no relation).


## Phase 14: full growth with widening for every convolution

Phase 12 repeated after the widening refactor (`91085d0`). Before it, widening was offered only where copied channels kept the output identical: in practice only the first convolution, so 1–2 of the 27–90 candidates per search. Now every convolution can be widened ×2 or ×4. The new width is carried through BatchNorm, residual sums and the classifier after global pooling. New channels start with random weights, and every layer that reads them starts with zero weights for them, so the output is unchanged after the action. Everything else matches Phase 12. Seeds 1–3.

| Run | Seed | Final validation | Final test | Final parameters | Search time |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Phase 14** | 1 | 63.68% | 64.65% | 51,448 | 113 min |
| **Phase 14** | 2 | Stopped after generation 8: 66.48% | — | 8,624 | 92 min |
| **Phase 14** | 3 | Not run | — | — | — |
| Phase 12 | 1 | 60.90% | 61.04% | 4,448 | 137 min |
| Phase 12 | 2 | 63.28% | 64.19% | 5,616 | 156 min |

![Phase 14 training history](/assets/experiments/018-widening-training.png)

Solid lines are validation accuracy, dashed lines training accuracy; dotted vertical lines mark generation boundaries, where the selected action is applied. The letters at the bottom name the action applied at each boundary (W widen convolution, SC sequential convolution with BN + ReLU, SC- without them, RC residual convolution, SL sequential linear, N add neurons, O other, e.g. a residual linear layer). Parameters use a log scale.


### How often widening was offered and selected

| Run | Seed | Widening candidates per search | Widening share of candidates | Times selected |
| --- | ---: | ---: | ---: | ---: |
| **Phase 14** | 1 | 5.3 | 18% | 2 of 9 |
| **Phase 14** | 2 | 6.8 | 25% | 2 of 8 |
| Phase 12 | 1 | 1.1 | 3% | 0 of 9 |
| Phase 12 | 2 | 1.6 | 4% | 0 of 9 |

### Decisions

Selected action type per generation, with parameters after the action and the change in validation accuracy over the next generation.

| Generation | Phase 14 seed 1 | Phase 14 seed 2 |
| ---: | --- | --- |
| 0 | Sequential convolution + BN (896, +16.4 pp) | Sequential convolution + BN (896, +8.7 pp) |
| 1 | Residual convolution (1,480, +2.7 pp) | Sequential convolution + BN (1,496, +6.2 pp) |
| 2 | Widen convolution (5,272, +3.2 pp) | Widen convolution (2,176, +1.4 pp) |
| 3 | Sequential linear (6,328, +1.7 pp) | Widen convolution (6,256, +1.6 pp) |
| 4 | Add neurons (7,704, -0.8 pp) | Sequential convolution + BN (6,856, +4.9 pp) |
| 5 | Dropout (7,704, +7.1 pp) | Sequential convolution + BN (7,456, +6.5 pp) |
| 6 | Sequential linear (8,760, +0.6 pp) | Residual convolution (8,040, +1.4 pp) |
| 7 | Widen convolution (14,520, +2.5 pp) | Residual convolution (8,624, +2.2 pp) |
| 8 | Sequential convolution without BN (51,448, +0.4 pp) | — |


## Phase 15a: which scoring setting predicts real training

Phase 13 found no relation between the simulation ranking and real training. Phase 15a reuses Phase 13's generation-3 model and its 16 candidates, whose real 20-epoch validation accuracy is known, and scores each of them in five settings: 5, 10 or 15 scoring epochs on a CCS simulation set of 10% or 25% of the training data. Every arm is measured 3 times with the same three training seeds, and the score is the mean of the last 2 scoring epochs, as in the search. The learning rate follows the restarted main schedule (`de231f4`). The best setting becomes the scoring of the next full-growth run.

| Scoring epochs | Simulation set | Rank correlation, 1 measurement | Rank correlation, mean of 3 | Choice: real rank | Choice: gain over control | Real best: score rank | Real top 5 in score top 5 | Measurement range | Seconds per measurement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 10% | 0.58 | 0.43 | 2 | +2.32 pp | 4 | 2 of 5 | 4.0 pp | 16 |
| 10 | 10% | 0.51 | 0.38 | 12 | -2.18 pp | 6 | 1 of 5 | 5.1 pp | 25 |
| 5 | 25% | 0.40 | 0.46 | 2 | +2.32 pp | 9 | 2 of 5 | 5.0 pp | 31 |
| 10 | 25% | 0.16 | 0.44 | 3 | +1.76 pp | 5 | 2 of 5 | 3.6 pp | 62 |
| 15 | 25% | 0.18 | 0.09 | 11 | -1.88 pp | 7 | 1 of 5 | 3.2 pp | 79 |

The control reached 54.68%. A rank correlation of 1 is the real order, 0 no relation. The measurement range is the mean difference between the highest and lowest of the three scores of one arm.

The best agreement is 5 scoring epochs on 25% of the training data (rank correlation 0.46 with 3 measurements).


## Phase 16: is BatchNorm or the residual block the missing piece?

The residual starter of the other branch (conv 16 → BatchNorm → ReLU, then relu(x + BatchNorm(conv(x)))) trained much better than our Experiment 010 starter, and our actions could not build two of its parts: a BatchNorm after an existing layer, and a residual block with BatchNorm. Phase 16 checks both. All parts are short Phase 14 runs (seed 1, CIFAR-10, the same search and scoring, 5 generations):

- **16a**: our starter with the new add-normalization action (`541cab3`) among the candidates: a BatchNorm after any convolution or linear layer that has none, with default initialization. Does the simulation score it high?
- **16b**: the same run from our starter with a BatchNorm already between conv1 and its ReLU, as in the residual starter's stem, with the same 8 channels and no residual block. If this run is much better than Phase 14, BatchNorm was what was missing; if not, the residual connection is needed as well.
- **16c**: the 16b starter plus the residual block of the other branch's starter after the stem, relu(x + BatchNorm(conv(x))), still 8 channels. Compared with 16b, it shows what the residual block adds.

### 16a: Exp 010 starter, normalization action

| Search after generation | Candidates | Normalization candidates | Best normalization score (rank) | Best score | Mean visits, normalization | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 6 | 1 | 33.4% (2 of 6) | 36.6% | 7.0 | Sequential convolution + BN |
| 1 | 14 | 1 | 41.3% (6 of 14) | 43.0% | 3.0 | Widen convolution |
| 2 | 13 | 1 | 39.8% (5 of 13) | 42.4% | 3.0 | Sequential convolution + BN |
| 3 | 21 | 1 | 45.2% (4 of 21) | 46.7% | 3.0 | Sequential convolution + BN |

![Phase 16a candidate scores](/assets/experiments/018-normalization-16a-scores.png)

Each dot is one measured candidate of one search; normalization candidates are magenta and the selected candidate has a black ring.

### 16b: Starter with BatchNorm after conv1

| Search after generation | Candidates | Normalization candidates | Best normalization score (rank) | Best score | Mean visits, normalization | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 5 | 0 | — | 34.2% | — | Sequential convolution + BN |
| 1 | 13 | 0 | — | 43.0% | — | Sequential convolution + BN |
| 2 | 23 | 0 | — | 47.3% | — | Sequential convolution + BN |
| 3 | 34 | 0 | — | 48.7% | — | Residual convolution |

![Phase 16b candidate scores](/assets/experiments/018-normalization-16b-scores.png)

Each dot is one measured candidate of one search; normalization candidates are magenta and the selected candidate has a black ring.

### 16c: Starter with BatchNorm and a residual block

| Search after generation | Candidates | Normalization candidates | Best normalization score (rank) | Best score | Mean visits, normalization | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 17 | 0 | — | 42.8% | — | Widen convolution |
| 1 | 17 | 0 | — | 47.3% | — | Sequential linear |
| 2 | 29 | 1 | 42.1% (24 of 29) | 49.9% | 3.0 | Sequential convolution + BN |
| 3 | 42 | 1 | 44.1% (33 of 42) | 52.2% | 3.0 | Add neurons |

![Phase 16c candidate scores](/assets/experiments/018-normalization-16c-scores.png)

Each dot is one measured candidate of one search; normalization candidates are magenta and the selected candidate has a black ring.

### Training

![Phase 16 training history](/assets/experiments/018-normalization-training.png)

Solid lines are validation accuracy, dashed lines training accuracy.

| Run | Epochs | Validation at the last epoch | Test | Parameters |
| --- | ---: | ---: | ---: | ---: |
| Phase 14 seed 1 | 100 | 53.78% | — | 6,328 |
| Phase 16a: Exp 010 starter, normalization action | 100 | 57.64% | 57.76% | 2,888 |
| Phase 16b: Starter with BatchNorm after conv1 | 100 | 58.24% | 58.47% | 2,696 |
| Phase 16c: Starter with BatchNorm and a residual block | 100 | 63.24% | 64.04% | 5,611 |


## Phase 17: does the search beat random growth on a long run?

Phase 16c showed that the residual block in the starter is worth about 5 pp. Phase 17 grows the same small starter for 15 generations of 20 epochs (300 epochs, as long as the residual-block experiments on main), with three arms on the same schedule and growth candidates (no dropout, deleting layers or neurons, or shrinking blocks):

- **17a**: Monte Carlo 2 with more search: 10 scoring epochs on the CCS 10% simulation set (the best setting in Phase 15a), 60 s per candidate (5-25 minutes per search), no search started after 7.5 hours. Seed 1.
- **17b**: the same candidates, one picked at random (a random action type, then a random action of that type). Seeds 1 and 2.
- **17c**: the starter trained for 300 epochs without growing. Seeds 1 and 2.

No experiment so far compared the search with random choice. If 17a does not beat 17b by more than the seed-to-seed spread (about 3 pp in Phase 12), the search does not pay for its time; 17c shows how much growth adds at all.

| Arm | Seed | Epochs | Validation at the last epoch | Test | Parameters | Selected actions |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 17a: Monte Carlo 2 search | 1 | 300 of 300 | 81.74% | 82.26% | 122,624 | Sequential convolution + BN 4, Widen convolution 4, Residual convolution 1 |
| 17b: random growth | 1 | 300 of 300 | 73.18% | 72.56% | 11,852 | AddSeqConv 2, AddNormalization 3, ResidualBlock 1, WidenConv 1, AddSeqLinear 2, AddResConv 1, AddNeurons 3, AddResLinear 1 |
| 17b: random growth | 2 | 300 of 300 | 85.56% | 85.65% | 5,388,544 | WidenConv 3, AddResConv 2, ResidualBlock 5, AddSeqLinear 1, AddNeurons 1, AddNormalization 1, AddSeqConv 1 |
| 17c: no growth | 1 | 300 of 300 | 53.94% | 53.43% | 904 | — |
| 17c: no growth | 2 | 300 of 300 | 52.60% | 53.22% | 904 | — |

![Phase 17 training history](/assets/experiments/018-long-run-training.png)

Solid lines are seed 1, dotted lines seed 2.

### 17a decisions

| Search after generation | Selected | Candidates | Mean measurements per candidate | Search time | Parameters after | Validation at the end of the next generation |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | Sequential convolution + BN — conv1 → linear2 / 8→8, 3×3 | 14 | 2.3 | 14 min | 1,504 | 48.3% |
| 1 | Widen convolution — block_conv → 32 ch | 19 | 1.9 | 19 min | 12,616 | 56.6% |
| 2 | Sequential convolution + BN — conv1 → block_conv / 32→32, 3×3 | 17 | 1.0 | 18 min | 21,928 | 64.1% |
| 3 | Widen convolution — seq_conv_0 → 32 ch | 23 | 1.0 | 55 min | 29,152 | 51.1% |
| 4 | Residual convolution — conv1 → seq_conv_0 / 32→32, 3×3 | 26 | 1.0 | 98 min | 38,400 | 66.5% |
| 5 | Sequential convolution + BN — conv1 → res_conv__0 / 32→32, 3×3 | 35 | 1.0 | 121 min | 47,712 | 70.0% |
| 6 | Sequential convolution + BN — seq_conv_2 → res_conv__0 / 32→32, 3×3 | 41 | 1.1 | 25 min | 57,024 | 72.2% |
| 7 | Widen convolution — block_conv → 64 ch | 48 | 1.0 | 28 min | 94,880 | 69.0% |
| 8 | Widen convolution — seq_conv_2 → 64 ch | 40 | 1.1 | 26 min | 122,624 | 75.6% |


## Phase 18: Monte Carlo 1 against Monte Carlo 2

Phase 17 stopped searching after 7.5 hours, so its search grew the network in only 9 of 14 generations. Phase 18 drops that deadline and compares the two search algorithms on the Phase 17 settings: the residual-block starter, 15 generations of 30 epochs (450 epochs), 10 scoring epochs on the CCS 10% simulation set, 60 s per candidate within 5-25 minutes per search, and growth actions only. Seeds 1 and 2 for each algorithm.

Both algorithms measure every candidate at least once before they stop, so a search can run past its 25 minutes when there are many candidates.

Pending. Run with `--phase mc1 --seeds 1 2` and `--phase mc2 --seeds 1 2`.
