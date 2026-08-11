# Experiment 004: Composed learning-rate schedules

We continue with the meta-parameters from Experiment 003 after the fix. The initial architecture is the `big` starter from Experiment 003. Only the learning-rate schedule changes.

Two questions:

1. Under real GrowingNN training with architecture actions, does each schedule behave as designed?
2. Which schedule gives stronger MNIST validation accuracy under that same setup?

Script: `experiments/train_mnist_exp004_composed_lr_schedulers.py`

Charts: `documentation/website/scripts/generate_experiment_004_charts.py`

Folder: `experiments/output/train_mnist/runs/exp004_composed_lr_schedulers`

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| LR schedule | `recovery_only_logistic`, `composed_cosine`, `composed_step`, `composed_exponential`, `composed_linear`, `composed_constant`, `composed_linear_1_to_0p1` | Compare recovery-only vs composed bases, plus custom cascade |
| Seed | `100`, `101`, `102` | Three matched seeds per schedule |

| Fixed parameter | Value |
| --- | ---: |
| Starter | `big` (`BigAvgPoolMnistNet`, channels `4`, hidden `16`) |
| Simulation grading | `val_acc` |
| Slope threshold | `3°` |
| Recovery warmup | logistic, `warmup_iterations=10`, `k=10` |
| Standard cell `lr_alpha` | `0.01` |
| Generations | `10` |
| Epochs per generation | `10` |
| Total training epochs | `100` |
| Simulation time | `120 s` |
| Cosine / standard linear `eta_min` | `0.001` |
| Step size / gamma | `33` / `0.5` |
| Exponential gamma | `0.98` |

For every `composed_*` schedule, the optimizer learning rate is the product of two functions:

```text
effective_lr = max(0.001, base_lr(global_epoch) * recovery_factor)
```

- `base_lr(global_epoch)` is the global schedule over the whole run. Examples: cosine decay, step drops, linear decay. It keeps moving with time even when no architecture action happens.
- `recovery_factor` is the GrowingNN action recovery. After an architecture change it starts near `0` and warms up to `1`. When no recent action is active, it stays at `1`.
- The `*` joins those two functions. Recovery scales the current base value down after an action, then lets it rise back to the base.
- `max(0.001, ...)` is a hard floor. The optimizer LR never goes below `0.001`. This experiment also checks whether that floor is too strict for some schedules.
- `effective_lr` is the value written into the optimizer and recorded in `training.json`.

`train_generations` calls `structure_changed()` only after an architecture action. That resets recovery. The next generation then starts near the floor and warms back toward the current base.

## Script and result timeline

All `21` boards completed (`7` schedules × `3` seeds).

Runtime from board metadata: `2026-08-10T22:11:58Z` to `2026-08-11T11:07:43Z`. Summed wall time across runs is about `15.7 hours`.

Every run recorded exactly `100` epochs.

## Why this experiment

Older MNIST runs mixed architecture change with one GrowingNN warmup schedule. That made it hard to see if the learning-rate curve itself was healthy.

Here the search setup stays fixed. Only the LR rule changes. A normal global curve can still run. Recovery still drops the LR after each architecture action and then raises it again.

We check whether the recorded LR series matches that design. We also compare final validation scores across schedules.

## What each schedule is

Short list of the seven cells. The strip below shows one demo action after generation `3` for each shape.

![Scheduler shape guide](/assets/experiments/004-scheduler-shape-guide.png)

> [!CAPTION] Figure 1. Quick shape guide. One small panel per schedule. Each panel has one demo architecture action after generation `3`.

- `recovery_only_logistic`: Exp 003 style. Absolute logistic warmup only. Peak stays near `0.01`. No global decay.
- `composed_cosine`: Cosine base from `0.01` to `0.001`, times logistic recovery.
- `composed_step`: Step base drops by `0.5` every `33` epochs, times logistic recovery.
- `composed_exponential`: Exponential base with gamma `0.98`, times logistic recovery.
- `composed_linear`: Linear base from `0.01` to `0.001`, times logistic recovery.
- `composed_constant`: Flat base `0.01`, times logistic recovery after actions.
- `composed_linear_1_to_0p1`: Custom cascade. Base goes from `1.0` to `0.1`, times logistic recovery.

## Do the schedulers work correctly?

### Measured learning-rate timelines

Each figure shows one schedule. Left panel is seed `100`. Right panel is seed `101`.

Top row: orange dashed line is the base LR. Blue line is the measured effective LR. Gray dotted lines mark epochs just after architecture actions.

Bottom row: green line is the recovery factor (`0` to `1`).

The y-axis for LR is scaled to that panel’s own range, so schedules near `0.01` stay readable.

#### Recovery-only logistic

This is the basic Exp 003 control. There is no global decay. The orange base is flat at `0.01`. The blue measured LR is the absolute logistic warmup. After an action it drops near the floor and climbs back to `0.01`.

![LR recovery-only logistic seeds 100 and 101](/assets/experiments/004-lr-recovery_only_logistic-seeds-100-101.png)

> [!CAPTION] Figure 2. `recovery_only_logistic`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Composed cosine

![LR composed cosine seeds 100 and 101](/assets/experiments/004-lr-composed_cosine-seeds-100-101.png)

> [!CAPTION] Figure 3. `composed_cosine`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Composed step

![LR composed step seeds 100 and 101](/assets/experiments/004-lr-composed_step-seeds-100-101.png)

> [!CAPTION] Figure 4. `composed_step`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Composed exponential

![LR composed exponential seeds 100 and 101](/assets/experiments/004-lr-composed_exponential-seeds-100-101.png)

> [!CAPTION] Figure 5. `composed_exponential`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Composed linear

![LR composed linear seeds 100 and 101](/assets/experiments/004-lr-composed_linear-seeds-100-101.png)

> [!CAPTION] Figure 6. `composed_linear`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Composed constant

![LR composed constant seeds 100 and 101](/assets/experiments/004-lr-composed_constant-seeds-100-101.png)

> [!CAPTION] Figure 7. `composed_constant`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

#### Custom cascade linear 1.0 to 0.1

![LR cascade 1.0 to 0.1 seeds 100 and 101](/assets/experiments/004-lr-composed_linear_1_to_0p1-seeds-100-101.png)

> [!CAPTION] Figure 8. `composed_linear_1_to_0p1`, seeds `100` and `101`. Base peak, measured LR, and recovery factor.

Composed schedules start at the full base value. Recovery is already warm until the first action. Recovery-only starts low and warms up to `0.01`, because that schedule is absolute warmup from the beginning.

The hard floor `0.001` is visible on late cosine, step, exponential, and linear bases. Those curves still look correct. The floor did not break the designed shapes in this grid.

## Training accuracy timelines

Same seed layout as the LR figures. Left is seed `100`. Right is seed `101`.

#### Recovery-only logistic

![Train acc recovery-only logistic seeds 100 and 101](/assets/experiments/004-train-acc-recovery_only_logistic-seeds-100-101.png)

> [!CAPTION] Figure 9. Training accuracy (%) for `recovery_only_logistic`. Left: seed `100`. Right: seed `101`.

#### Composed cosine

![Train acc composed cosine seeds 100 and 101](/assets/experiments/004-train-acc-composed_cosine-seeds-100-101.png)

> [!CAPTION] Figure 10. Training accuracy (%) for `composed_cosine`. Left: seed `100`. Right: seed `101`.

#### Composed step

![Train acc composed step seeds 100 and 101](/assets/experiments/004-train-acc-composed_step-seeds-100-101.png)

> [!CAPTION] Figure 11. Training accuracy (%) for `composed_step`. Left: seed `100`. Right: seed `101`.

#### Composed exponential

![Train acc composed exponential seeds 100 and 101](/assets/experiments/004-train-acc-composed_exponential-seeds-100-101.png)

> [!CAPTION] Figure 12. Training accuracy (%) for `composed_exponential`. Left: seed `100`. Right: seed `101`.

#### Composed linear

![Train acc composed linear seeds 100 and 101](/assets/experiments/004-train-acc-composed_linear-seeds-100-101.png)

> [!CAPTION] Figure 13. Training accuracy (%) for `composed_linear`. Left: seed `100`. Right: seed `101`.

#### Composed constant

![Train acc composed constant seeds 100 and 101](/assets/experiments/004-train-acc-composed_constant-seeds-100-101.png)

> [!CAPTION] Figure 14. Training accuracy (%) for `composed_constant`. Left: seed `100`. Right: seed `101`.

#### Custom cascade linear 1.0 to 0.1

![Train acc cascade seeds 100 and 101](/assets/experiments/004-train-acc-composed_linear_1_to_0p1-seeds-100-101.png)

> [!CAPTION] Figure 15. Training accuracy (%) for `composed_linear_1_to_0p1`. Left: seed `100`. Right: seed `101`.

## Validation accuracy timelines

Same seed layout as above.

#### Recovery-only logistic

![Val acc recovery-only logistic seeds 100 and 101](/assets/experiments/004-val-acc-recovery_only_logistic-seeds-100-101.png)

> [!CAPTION] Figure 16. Validation accuracy (%) for `recovery_only_logistic`. Left: seed `100`. Right: seed `101`.

#### Composed cosine

![Val acc composed cosine seeds 100 and 101](/assets/experiments/004-val-acc-composed_cosine-seeds-100-101.png)

> [!CAPTION] Figure 17. Validation accuracy (%) for `composed_cosine`. Left: seed `100`. Right: seed `101`.

#### Composed step

![Val acc composed step seeds 100 and 101](/assets/experiments/004-val-acc-composed_step-seeds-100-101.png)

> [!CAPTION] Figure 18. Validation accuracy (%) for `composed_step`. Left: seed `100`. Right: seed `101`.

#### Composed exponential

![Val acc composed exponential seeds 100 and 101](/assets/experiments/004-val-acc-composed_exponential-seeds-100-101.png)

> [!CAPTION] Figure 19. Validation accuracy (%) for `composed_exponential`. Left: seed `100`. Right: seed `101`.

#### Composed linear

![Val acc composed linear seeds 100 and 101](/assets/experiments/004-val-acc-composed_linear-seeds-100-101.png)

> [!CAPTION] Figure 20. Validation accuracy (%) for `composed_linear`. Left: seed `100`. Right: seed `101`.

#### Composed constant

![Val acc composed constant seeds 100 and 101](/assets/experiments/004-val-acc-composed_constant-seeds-100-101.png)

> [!CAPTION] Figure 21. Validation accuracy (%) for `composed_constant`. Left: seed `100`. Right: seed `101`.

#### Custom cascade linear 1.0 to 0.1

![Val acc cascade seeds 100 and 101](/assets/experiments/004-val-acc-composed_linear_1_to_0p1-seeds-100-101.png)

> [!CAPTION] Figure 22. Validation accuracy (%) for `composed_linear_1_to_0p1`. Left: seed `100`. Right: seed `101`.

## Final accuracy by schedule

![Final accuracy by schedule](/assets/experiments/004-final-accuracy-by-schedule.png)

> [!CAPTION] Figure 23. Mean final training and validation accuracy by schedule. Dots are individual seeds. Diamonds are validation; circles are training.

| Schedule | Mean final training | Mean final validation | Best final validation | Worst final validation | Mean peak validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| `composed_step` | `82.61%` | `89.37%` | `92.11%` | `87.89%` | `89.65%` |
| `composed_exponential` | `87.25%` | `88.66%` | `90.76%` | `86.69%` | `89.65%` |
| `composed_cosine` | `81.60%` | `86.38%` | `90.40%` | `81.90%` | `87.75%` |
| `composed_constant` | `78.27%` | `82.80%` | `95.31%` | `73.95%` | `83.74%` |
| `recovery_only_logistic` | `68.06%` | `74.53%` | `86.87%` | `50.37%` | `77.16%` |
| `composed_linear` | `62.19%` | `64.05%` | `92.76%` | `11.03%` | `70.57%` |
| `composed_linear_1_to_0p1` | `18.11%` | `15.40%` | `25.07%` | `10.57%` | `42.23%` |

Top three on mean final validation:

- `composed_step` is best on the mean (`89.37%`) and has the tightest seed band.
- `composed_exponential` is very close (`88.66%`) and looks very stable on the timelines. In large-model training, exponential and cosine decays are the common defaults. When step and exponential are this close, prefer `composed_exponential` as the default GrowingNN package.
- `composed_cosine` is also strong (`86.38%`). It is a standard PyTorch-style choice, a little behind step and exponential here.

`composed_constant` can hit the single best seed (`95.31%`), but the mean is lower and less stable. Recovery-only is weaker. Standard linear has one collapse. The custom cascade is far too large for this MNIST SGD setup.

## Train-accuracy change after architecture actions

This section follows the Experiment 003 idea: after each architecture action, give the model one generation to recover, then measure the change in training accuracy. Values are percentage points. Negative values mean the run lost training accuracy over that recovery window.

![Post-action train accuracy change by schedule](/assets/experiments/004-post-action-train-acc-change-by-schedule.png)

> [!CAPTION] Figure 24. Train-accuracy change one generation after each architecture action. One panel per LR schedule. Dots are actions. The bar is the mean.

| Schedule | Observed actions | Mean train change | Negative share |
| --- | ---: | ---: | --- |
| `composed_exponential` | `22` | `+7.49` percentage points | `1/22` |
| `composed_cosine` | `22` | `+6.76` percentage points | `0/22` |
| `composed_constant` | `23` | `+6.72` percentage points | `3/23` |
| `composed_step` | `18` | `+6.65` percentage points | `2/18` |
| `recovery_only_logistic` | `22` | `+5.82` percentage points | `1/22` |
| `composed_linear` | `20` | `+4.56` percentage points | `4/20` |
| `composed_linear_1_to_0p1` | `20` | `-3.18` percentage points | `7/20` |

Cosine and exponential lose almost no training accuracy after actions. Step is also strong, with only two negative events. Linear and especially the cascade lose training accuracy more often. So a high final score is not enough: the cascade schedule both finishes poorly and often drops train accuracy after mutations.

## Conclusions

1. The schedulers work as designed. Measured LR follows the base curve, and recovery dips appear after architecture actions.
2. The hard floor `0.001` did not break the designed shapes. Late cosine, step, exponential, and linear bases sit on that floor as expected.
3. Prefer `composed_exponential` as the default package. It is nearly as strong as step on mean final validation, more standard in large-model training, and almost never loses train accuracy after actions.
4. Avoid the custom cascade at `1.0→0.1` for this MNIST SGD setup. The curve is correct, but the scale is too high and post-action train losses are common.

## Next experiments

This LR-composition study is done. There is no strong follow-up experiment required from these results. Later work can use `composed_exponential` as the fixed LR package and move on to other GrowingNN topics.
