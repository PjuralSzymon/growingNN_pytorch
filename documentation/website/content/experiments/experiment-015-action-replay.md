---
title: "Experiment 15: replaying the epoch-80 decision"
---

All arms start from the Experiment 13 generation-3 pre-decision checkpoint, after epoch 80. Each recorded action and an unchanged control is trained for 20 epochs with the same nominal seed, training recipe and saved learning-rate schedule. The primary comparison uses 3,880 validation examples outside the original 1,120-example simulation validation subset.

The five actions and the original simulation scores are from the recorded decision. Resetting the seed does not guarantee identical random draws across different architectures. This is one checkpoint and one training seed; it is not a multi-seed estimate.

| Action | Original simulation | Parameters | Initial held-out | After 1 epoch | After 20 epochs | Difference vs control | Full validation after 20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | — | 2,368 | 33.58% | 36.70% | 38.25% | +0.00 pp | 38.52% |
| Widen to 256 | 37.32% | 9,472 | 33.61% | 37.73% | 40.05% | +1.80 pp | 40.12% |
| Widen to 128 | 36.07% | 4,736 | 33.61% | 38.99% | 36.86% | -1.39 pp | 37.08% |
| Add linear | 34.91% | 6,528 | 33.58% | 31.93% | 41.91% | +3.66 pp | 41.64% |
| Conv without BN/ReLU | 25.98% | 39,296 | 33.58% | 28.48% | 39.90% | +1.65 pp | 39.70% |
| A08: conv + BN + ReLU | 39.15% | 39,424 | 33.58% | 45.00% | 56.80% | +18.56 pp | 56.80% |

![Held-out validation accuracy for all replay arms](/assets/experiments/015-action-replay-validation.png)

Full validation includes the original simulation-validation subset; held-out validation is the primary comparison. No test-set scores were used. The original run followed a different random trajectory, so its subsequent accuracy need not be reproduced exactly.


## Phase 2: scoring epochs

Each cell is validation accuracy after scoring a fresh copy of the same action, using the original simulation subset and the Monte Carlo rollout scoring path with rollout depth zero. The epoch settings are separate runs, using the same nominal seed; one score per action is not a stability estimate.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 36.43% | 39.20% | 32.50% | 29.55% | 34.82% |
| 5 | 39.29% | 38.66% | 39.38% | 29.91% | 48.30% |
| 10 | 37.14% | 38.21% | 41.96% | 36.70% | 41.43% |
| 20 | 37.23% | 40.00% | 40.00% | 36.79% | 48.57% |

![Simulation accuracy by scoring epochs and action](/assets/experiments/015-scoring-epochs.png)

These simulation scores use different data and training budgets than the full-training results above.


## Phase 3: Monte Carlo 2 search scores

Each row runs the complete Monte Carlo 2 search from the same checkpoint with the five recorded root actions. Cells show the final root-node mean search value and visit count; **bold** marks the selected action. These values combine rollout scores and are not necessarily direct validation accuracies. The search can include randomly chosen future actions. The configured 32-second budget is soft and may be exceeded while completing search.

| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3496 (n=1) | **0.3695 (n=1)** | 0.3293 (n=1) | 0.2540 (n=1) | 0.3429 (n=1) |
| 5 | 0.3777 (n=1) | 0.3776 (n=1) | 0.3316 (n=1) | 0.2621 (n=1) | **0.4311 (n=1)** |
| 10 | 0.3791 (n=1) | 0.3952 (n=1) | 0.3888 (n=1) | 0.3386 (n=1) | **0.4237 (n=1)** |
| 20 | 0.3996 (n=1) | 0.3986 (n=1) | 0.3713 (n=1) | 0.2504 (n=1) | **0.4835 (n=1)** |

![Monte Carlo 2 search values by scoring epochs](/assets/experiments/015-montecarlo2-epochs.png)

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

![Monte Carlo 2 search values by time budget](/assets/experiments/015-montecarlo2-time.png)

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

![Seed 1 training accuracy and model parameters](/assets/experiments/015-full-seed1-training.png)

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

![Seed 1 candidate score distributions](/assets/experiments/015-full-seed1-scores-overview.png)

Individual action charts: [generation 0](/assets/experiments/015-full-seed1-scores-gen-0.png), [generation 1](/assets/experiments/015-full-seed1-scores-gen-1.png), [generation 2](/assets/experiments/015-full-seed1-scores-gen-2.png), [generation 3](/assets/experiments/015-full-seed1-scores-gen-3.png), [generation 4](/assets/experiments/015-full-seed1-scores-gen-4.png), [generation 5](/assets/experiments/015-full-seed1-scores-gen-5.png), [generation 6](/assets/experiments/015-full-seed1-scores-gen-6.png), [generation 7](/assets/experiments/015-full-seed1-scores-gen-7.png), [generation 8](/assets/experiments/015-full-seed1-scores-gen-8.png).

[Download all recorded candidate scores and full action descriptions](/assets/experiments/015-full-seed1-candidates.csv).

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

This phase repeats the Phase 3 search on the same Experiment 013 generation-3 checkpoint and five recorded root actions, with the fixed algorithm and the settings planned for Phase 7: **rollout depth 0**, **5 scoring epochs**, convolution widening enabled, UCB C=1.0, minimum iterations 3 and a 32-second soft budget. Three search seeds are run to show how stable the ranking is. Cells show the mean search value and visit count; **bold** marks the selected action.

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

![Phase 7 and Phase 5 seed 1 validation accuracy and parameters](/assets/experiments/015-phase7-seed1-vs-phase5.png)

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
| — | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Run the experiment script with `--phase check-noise-fix --seeds 1 2 3`, then regenerate this report.
