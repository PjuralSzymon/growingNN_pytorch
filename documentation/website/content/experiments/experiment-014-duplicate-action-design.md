# Experiment 014: duplicate-action design

## Summary

- **Everything here is CIFAR-10.** The `off` step is **not** a model without growth. It disables only the residual-block actions; all other actions (residual conv, seq conv, neurons, dropout, and others) still grow the network every generation.
- **`off` is high (72.4% validation) because of the starting network and the longer run, not because of this branch.** Experiment 014 reuses the Experiment 013 setup: a 2,970-parameter starter with batch norm and a residual block, 29 searches, and 300 epochs. Experiment 010 started from a 296-parameter starter with no batch norm, 9 searches, and 200 epochs. Experiment 013 already measured 73.0% with block actions off on this setup (Figure 1).
- **The redesigned duplicate helps when it is applied.** On the same checkpoint, inserting the copy after the block's ReLU raised validation accuracy by **+4.2 pp** over no change after one generation. The original duplicate gave **+1.9 pp** (Figure 2).
- **It is offered more often but not chosen more often** (Figures 8–9). With ~150 candidates per search and simulation-score noise of SD ≈ 11.7 pp, a 2 pp advantage after one epoch is invisible to the search.
- **Final accuracies in the growing runs are seed noise.** The best step, `+after_relu`, never executed a residual-block action. The two `off` seeds differ by 12.8 pp (Figure 5, Table 3).

## Background: why `off` reaches ~72% here but Experiment 010 stayed below 60%

This branch changed only the residual-block action. The higher accuracy comes from the experiment setup, which Experiment 014 copies from Experiment 013:

| | Experiment 010 (MCTS meta-parameters) | Experiment 013 / 014 (this branch) |
| --- | --- | --- |
| Dataset | CIFAR-10 | CIFAR-10 |
| Starter network | `Conv 3→8` + ReLU + pool + `Linear 8→10` | `Conv 3→16` + BN + ReLU, residual block (`Conv 16→16` + BN, add, ReLU), pool + `Linear 16→10` |
| Starter parameters | **296** | **2,970** (10×) |
| Batch norm / residual at start | No / no | Yes / yes |
| Schedule | 10 generations × 20 epochs = 200 epochs | 30 generations × 10 epochs = 300 epochs |
| Architecture searches (one after each generation except the last) | 9 | 29 |
| Optimizer, LR, augmentation | SGD 0.9, cosine 0.1 → 0.001, cutout, label smoothing 0.1 | Same |
| Search (Exp014) | Tuned: 32 s, 10% sim set, depth 1 | Old defaults: 8 s, 128 sim examples |
| Final parameters | — | 12,556–65,802 |

> [!CAPTION] Table 1. Setup differences between Experiment 010 and Experiments 013/014. The training recipe is the same; the starter, schedule length, and number of searches differ.

![Validation accuracy of Experiment 010, 013 and 014 setups](/assets/experiments/014-context.png)

> [!CAPTION] Figure 1. CIFAR-10 validation accuracy, mean of seeds, at the end of each setup's budget. Exp010 and Exp013 values are taken from their reports. The third bar is Experiment 014 `off` stopped at epoch 200, the same epoch budget as Experiment 010.

Three factors explain the gap:

1. **Starter.** A 16-channel network with batch norm and a residual connection trains much faster than an 8-channel network with no batch norm. At epoch 200, the budget of Experiment 010, `off` already reaches 68.5%, compared with 59.2% for the tuned Experiment 010 search.
2. **More searches and epochs.** 29 searches add 29 components (Figure 10), compared with 9 in Experiment 010, and the run trains for 100 more epochs.
3. **The search is not better here.** Experiment 014 uses the old 8 s / 128-example search, which scored 46.7% in Experiment 010. The improvement comes from the starting point, not from MCTS.

So `off` is the correct baseline for this experiment, and the comparisons are against it. It should not be compared directly with Experiment 010 numbers. To transfer the result, rerun Experiment 010's tuned search with this residual starter, or this experiment with the 296-parameter starter.

## Question and design

Do four changes to the residual-block duplicate action make it more useful, and does MCTS choose it more often? The changes are applied cumulatively:

| Step | Change |
| --- | --- |
| off | Residual-block actions disabled; all other actions enabled |
| current | Original action: quasi-identity copy before the ReLU; strict block discovery; hard remove |
| +side | Blocks with side inputs (for example from AddResConv) remain discoverable |
| +after_relu | The copy is inserted after the block's ReLU: `relu(x + F(x))` → copy |
| +scale | Residual scale ε on the copy's last layer; Part A selected ε = 1.0, so this step repeats `+after_relu` |
| +merge | Shrink folds a block into an identical neighbour instead of deleting it |

> [!CAPTION] Table 2. Steps of the ladder. Each step keeps every change from the steps above it.

- **Part A (same-checkpoint test):** apply one action to a fixed trained checkpoint, then train one generation (10 epochs) with 2 seeds. Training is identical with no search, so the effect of the action is isolated.
- **Part B (growing runs):** Experiment 013 CIFAR-10 protocol (Table 1), seeds 1 and 2.

Script: **experiments/train_cifar10_exp014_duplicate_action_design.py**  
Run data: **experiments/output/train_cifar10/runs/exp014_duplicate_action_design/**  
Chart/snapshot generator: **documentation/website/scripts/generate_experiment_014_report.py**

## Part A: the action itself

![Validation gain over no change after one generation, per duplicate variant](/assets/experiments/014-paired-ladder.png)

> [!CAPTION] Figure 2. Validation accuracy after one generation minus the no-change result from the same checkpoint. Bars show the mean of 2 seeds; markers show individual seeds. Left: plain one-block checkpoint. Right: a checkpoint where AddResConv wired the stem into the block; the original action cannot find that block.

On the plain checkpoint, placing the copy after the ReLU doubles the gain (+1.9 → +4.2 pp). Side-input discovery makes the block available on the right panel, but the gain there is small (+0.8 pp). The copy also causes a 9–14 pp immediate drop, because it re-adds the shared side input.

![Validation accuracy over time after each action, grow and shrink](/assets/experiments/014-paired-trajectory.png)

> [!CAPTION] Figure 3. Validation accuracy at the checkpoint, right after the action, after 1 epoch, and after one generation (10 epochs); mean of 2 seeds. Left: duplicate variants on the plain checkpoint (`+scale 0.3` is the scale used before the sweep). Right: removing or merging one block of a two-block checkpoint.

The duplicate costs 4.4 pp at first, recovers after one epoch, and pulls ahead by the end of the generation. Most of its benefit appears **after** the first epoch, but the search scores candidates after one epoch.

![Effect of the residual scale on the duplicate](/assets/experiments/014-scale-sweep.png)

> [!CAPTION] Figure 4. Validation accuracy vs no change for the `+side +after_relu` duplicate at five residual scales ε, measured at three times. ε = 0 is an exact no-op copy; ε = 1 is the unscaled quasi-identity copy.

A larger ε hurts more immediately but trains better. ε = 1.0 is best after one generation (+4.0 pp); the exact-preservation setting ε = 0 reaches +2.8 pp. After one epoch, the point at which the search scores the action, the gain is only +1.7 pp at best.

| Shrink (two-block checkpoint) | Immediately | After 1 epoch | After one generation vs no change |
| --- | ---: | ---: | ---: |
| remove | −43.8 pp | 38.8% | −12.9 pp |
| merge | −27.2 pp | 51.3% | −9.5 pp |

> [!CAPTION] Table 3. Shrink options on the two-block checkpoint, mean of 2 seeds. No change reaches 59.1% after 1 epoch and 59.7% after one generation.

## Part B: growing runs

![Final test accuracy for each step](/assets/experiments/014-final-test.png)

> [!CAPTION] Figure 5. Final test accuracy after 300 epochs. Bars show the mean of 2 seeds; circles show seed 1 and squares seed 2.

| Step | Seed 1 test | Seed 2 test | Mean test | Mean val | Mean params | Block actions executed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| off | 78.72% | 65.89% | 72.30% | 72.36% | 35,835 | — |
| current | 72.73% | 71.87% | 72.30% | 72.78% | 41,970 | 2 duplicates (s1) |
| +side | 58.76% | 77.86% | 68.31% | 67.80% | 47,222 | 1 remove (s1), 1 duplicate (s2) |
| +after_relu | 79.82% | 73.68% | 76.75% | 76.50% | 65,786 | none |
| +scale (1.0) | 71.89% | 73.28% | 72.58% | 72.52% | 49,522 | 1 duplicate (s2) |
| +merge | 72.33% | 80.95% | 76.64% | 76.22% | 54,434 | 1 duplicate (s2) |

> [!CAPTION] Table 4. Final results per step and seed. Validation and test are measured after the last epoch.

The spread between seeds (up to 19 pp) is larger than any difference between steps. The highest mean belongs to a step that never executed a block action, and `+scale` has the same settings as `+after_relu` but scored 4 pp lower. These finals cannot rank the designs.

![Validation accuracy histories for each step](/assets/experiments/014-val-histories.png)

> [!CAPTION] Figure 6. Validation accuracy per epoch. Solid = seed 1, dashed = seed 2; the light gray line is the mean of the `off` seeds. Vertical lines mark executed residual-block actions: green = duplicate, red = remove.

![Parameter count histories for each step](/assets/experiments/014-param-histories.png)

> [!CAPTION] Figure 7. Parameter count per epoch, with the same encoding as Figure 6. Growth in every step, including `off`, comes mostly from residual conv and seq conv layers.

The early duplicates (generations 0–5) do not separate from `off`. In `current` seed 1, the generation-5 duplicate dropped validation from 55.9% to 41.1%. In `+side` seed 1, removing the only block in generation 1 (39.5% → 26.9%) left that run about 15 pp below the others for the remaining 28 generations. The `off` seed 2 run grew to only 12,556 parameters and finished 12.8 pp below seed 1 (59,114 parameters).

### Is the duplicate chosen more often?

![How often the duplicate is offered, ranked in the top 5, and chosen](/assets/experiments/014-duplicate-selection.png)

> [!CAPTION] Figure 8. Counts over 58 MCTS searches per step (29 per seed). "Offered" means at least one duplicate candidate existed; "top 5" means the best duplicate ranked in the top 5 of the search's candidates.

![Duplicate simulation score compared with no change and with the winning candidate](/assets/experiments/014-score-gaps.png)

> [!CAPTION] Figure 9. Per search, the score of the best duplicate candidate minus the median dropout score (left) and minus the chosen candidate's score (right). Dropout does not change eval predictions, so its score serves as the no-change score. Box = interquartile range; dots = searches; gray band = ±1 SD of the dropout scores within a search.

| Step | Offered | Median score vs no change | Median gap to winner | Median rank (of ~150) | Chosen |
| --- | ---: | ---: | ---: | ---: | ---: |
| current | 22/58 | +1.7 pp | −10.0 pp | 15 | 2 |
| +side | 47/58 | −3.3 pp | −13.3 pp | 35 | 1 |
| +after_relu | 33/58 | −1.7 pp | −10.0 pp | 21 | 0 |
| +scale (1.0) | 35/58 | −6.7 pp | −16.7 pp | 48 | 1 |
| +merge | 39/58 | −3.3 pp | −16.7 pp | 34 | 1 |

> [!CAPTION] Table 5. Duplicate selection statistics over 58 searches per step (searches where a duplicate was offered).

Availability improved with side-input discovery. Selection did not. The duplicate scores about the same as no change, inside the ±11.7 pp noise band, and the winner of each search, the luckiest of ~150 candidates, scores about 10–17 pp above it.

![Executed action types per step](/assets/experiments/014-action-mix.png)

> [!CAPTION] Figure 10. Executed actions by type, summed over both seeds (58 actions per step). Residual conv and seq conv dominate every step; residual-block actions (red) are 0–2 per step.

## Limitations

- Two seeds per step; the seed spread (up to 19 pp) is larger than the effects being measured.
- Only 6 residual-block actions ran across 10 growing runs, and no merge ran, so Part B says almost nothing about the action's effect inside growth.
- `+scale` repeats `+after_relu` because the sweep selected ε = 1.0. The two steps measure run-to-run noise, not a scale effect.
- Part A uses small one- and two-block checkpoints at ~50–60% validation. Gains may differ on larger, later-stage models.
- Comparisons with Experiment 010 are across different starters, schedules, and search settings (Table 1). They explain the level of accuracy, not the effect of any single factor.

## Conclusions

1. **The higher accuracy on this branch comes from the Experiment 013 setup** (residual 2,970-parameter starter, 300 epochs, 29 searches), not from the residual-block action.
2. **Keep `after_activation=True` and `allow_side_inputs=True` with ε = 1.0** as the duplicate defaults. In the controlled test they give the largest per-generation gain (+4.2 pp vs +1.9 pp).
3. **The selection problem is in scoring, not in the action.** A ~2 pp gain after one epoch cannot beat ~150 candidates with ~12 pp score noise. This finding is for the simulation work: use more evaluation examples or visits per candidate, or score after a longer rollout.
4. **Merge is a softer shrink than remove,** but both cost 9–13 pp after a generation, so neither should replace the current shrink default yet.

## Follow-up analysis of the searches

- **Per candidate, MCTS favours the duplicate.** Across all 348 searches, one duplicate candidate won 2.0% of the time (5 of 254), compared with 0.9% for residual conv, 0.7% for seq conv, and 0.3% for dropout. The duplicate is rarely executed because it has about 1.5 candidates per search, against 53 for residual conv.
- **The final MCTS choice includes the exploration bonus.** `get_action` returns `root.get_best_child()`, which ranks children by UCB1 (`value + C · ln(N) / visits`) rather than by mean score. When visit counts differ, the least-visited child gets a larger bonus. In 13 of 348 searches the chosen candidate was not the top-scored one. The one executed block remove (`+side` seed 1, generation 1) scored 0.333 (4th of 12) but was chosen with UCB 6.22 after one visit, while the top-scored duplicate (0.467, two visits) had UCB 3.88. This belongs to the simulation work: the final choice should use mean score or visit count.

## Changes to the action after this experiment

The action was simplified to one version, and the compared design switches were removed from `growingnn` (the Experiment 014 script now runs only `--summary`):

- The duplicate is inserted after the block's ReLU and includes its own ReLU, with plain quasi-identity weights (ε = 1, no scale parameter).
- Blocks with side inputs are always discoverable, and **the copy no longer includes the side input**: it computes only `y + F'(y)`.
- Shrink: when two identical blocks are adjacent, they are merged; otherwise a block is removed. **The last residual block in the network is never removed.**

## Next experiments

- Run Experiment 010's tuned search with the residual starter (or this ladder with the 296-parameter starter), so the starter and search effects can be separated.
- Repeat Part A for the simplified action, also checking the side-input checkpoint, where the copy no longer doubles the side input.
- Repeat Part A on later-stage checkpoints from Part B (for example generation 15), to check whether the +4 pp gain holds on larger models.
- After the simulation changes land, rerun only `off`, `current`, and `+after_relu` with 3+ seeds.
