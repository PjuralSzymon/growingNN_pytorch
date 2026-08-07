# Experiment 003: Simulation grading metric

Experiment 002 showed that sequential convolution unlocked medium/small growth, but also that dropout was over-selected. Dropout can raise validation while training stays flat or falls. Simulation graded candidates by validation accuracy, so search could prefer dropout even when learning was weak.

This experiment asks one question: does grading simulation candidates by training accuracy reduce that dropout bias and improve growth, compared with grading by validation accuracy?

Script: `experiments/train_mnist_exp003_score_accuracy_metric.py`

Raw output:

`experiments/output/train_mnist/runs/exp003_score_accuracy_metric`

Runtime from board metadata: `2026-08-06T18:11:42Z` to `2026-08-06T22:05:37Z`. Recorded training time across the `16` completed runs is about `5.4 hours`.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation grading metric | `val_acc`, `train_acc` | Tests the Exp 002 scoring suspicion |
| Initial architecture | `big`, `medium_1conv_2linear` | Strongest corrected Exp 002 starters |
| Seed | `100`, `101`, `102`, `103` | Four matched seeds |

| Fixed parameter | Value |
| --- | ---: |
| Slope threshold | `3°` |
| LR warmup | logistic |
| Channels | `4` |
| Hidden linear size | `16` |
| Pooling | `adaptive_avg_pool2d` |
| Generations | `5` |
| Epochs per generation | `10` |
| Simulation time | `120 s` |

Config switch: `SimulationScore(accuracy_metric=...)` via `score_accuracy_metric`. Default for older experiments remains `val_acc`.

All `16` boards completed.

## Final accuracy by grading mode

![Mean final accuracy by grading mode and starter](/assets/experiments/003-final-accuracy-by-score-metric.png)

> [!CAPTION] Figure 1. Mean final training and validation accuracy. Chart labels: `grade val` = `val_acc`, `grade train` = `train_acc`, `med 1c+2l` = `medium_1conv_2linear`. Dots are per-seed finals.

| Grading mode | Starter | Mean final training | Mean final validation | Best final validation | Worst final validation | Mean dropout actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `val_acc` | `big` | `58.69%` | `58.58%` | `88.39%` | `27.57%` | `1.00` |
| `val_acc` | `medium_1conv_2linear` | `52.61%` | `58.27%` | `78.12%` | `25.40%` | `1.00` |
| `train_acc` | `big` | `76.47%` | `76.23%` | `89.57%` | `60.28%` | `0.25` |
| `train_acc` | `medium_1conv_2linear` | `42.55%` | `47.23%` | `77.21%` | `20.57%` | `2.25` |

The low dots in Figure 1 are seed collapses, not soft noise. Under `val_acc`, `big` has two weak finals (`39.40%`, `27.57%`). Under `train_acc`, `medium_1conv_2linear` has two weak finals (`20.57%`, `21.33%`). Under `train_acc`, `big` has no collapse: worst final validation is still `60.28%`.

Training grading helps `big` on the mean and removes the low outliers. It does not help `medium_1conv_2linear`: the mean falls because two seeds collapse. The Seed-level reading section ties each collapse to early sequential dropout.

## Dropout and action composition

![Mean sequential-dropout actions by grading mode](/assets/experiments/003-dropout-actions-by-score-metric.png)

> [!CAPTION] Figure 2. Mean sequential-dropout actions per completed seed. Dots are individual seeds.

![Executed action counts by type and grading mode](/assets/experiments/003-action-composition-by-score-metric.png)

> [!CAPTION] Figure 3. Total executed actions by type across all seeds of each grading mode.

| Action type | Count under `val_acc` | Count under `train_acc` |
| --- | ---: | ---: |
| Add sequential dropout | `8` | `10` |
| Add residual convolution | `5` | `7` |
| Add sequential linear | `7` | `3` |
| Add sequential convolution | `3` | `2` |
| Add residual linear | `1` | `0` |
| Delete layer | `0` | `1` |

Training grading does not globally remove dropout. Total dropout actions even rise from `8` to `10`. The change is uneven:

- on `big`, mean dropout falls from `1.00` to `0.25`
- on `medium_1conv_2linear`, mean dropout rises from `1.00` to `2.25`

Residual convolution rises under `train_acc` (`5` → `7`), which matches the stronger big mean.

## Training histories

![Training-accuracy curves by grading mode and starter](/assets/experiments/003-training-curves.png)

> [!CAPTION] Figure 4. Training curves for each grading mode × starter. Line color marks the seed.

### `val_acc` + `big`

Two strong seeds (`100`, `102`) start without dropout and reach the high-`70%` to high-`80%` validation range. Two weak seeds (`101`, `103`) start with sequential dropout and stay low (`39.40%`, `27.57%`). This repeats the Exp 002 collapse pattern.

### `train_acc` + `big`

All four seeds stay useful. Mean validation rises to `76.23%`. Best seed `103` reaches `89.57%` with two residual convolutions and no dropout. Only seed `101` uses one late dropout.

### `val_acc` + `medium_1conv_2linear`

Mixed but mostly alive. Best seed `100` reaches `78.12%` with no dropout. Seed `101` collapses after two early dropouts (`25.40%`).

### `train_acc` + `medium_1conv_2linear`

Two seeds collapse hard: `101` and `102` each execute four sequential dropouts and finish near `21%` validation with no parameter growth. Seed `100` is still strong (`77.21%`). So training grading makes medium less stable, not more stable.

## Seed-level reading

Collapsed runs (`final validation < 40%`):

| Grading mode | Starter | Seed | Final validation | First actions |
| --- | --- | ---: | ---: | --- |
| `val_acc` | `big` | `101` | `39.40%` | dropout, then seq conv / dropout |
| `val_acc` | `big` | `103` | `27.57%` | dropout first |
| `val_acc` | `medium_1conv_2linear` | `101` | `25.40%` | two dropouts first |
| `train_acc` | `medium_1conv_2linear` | `101` | `20.57%` | four dropouts |
| `train_acc` | `medium_1conv_2linear` | `102` | `21.33%` | four dropouts |

Strong runs under both modes still share the same useful pattern: residual convolution without early stacked dropout.

## Conclusions

1. Switching simulation grading from `val_acc` to `train_acc` is not a pure fix for the Exp 002 dropout problem.
2. For `big`, `train_acc` grading is clearly better: mean validation rises from `58.58%` to `76.23%`, mean dropout falls from `1.00` to `0.25`, and no seed fully collapses.
3. For `medium_1conv_2linear`, `train_acc` grading is worse: mean validation falls from `58.27%` to `47.23%`, and two seeds stack four dropouts with almost no learning.
4. Total dropout actions across the grid do not fall under `train_acc` (`8` → `10`). The bias moves between starters instead of disappearing.
5. Residual convolution remains the high-value action. Strong seeds under both modes use it and avoid early stacked dropout.
6. Action scoring alone is not enough. The next control should constrain early stacked dropout directly, or use a score that mixes training and validation instead of replacing one with the other.

## Next steps

1. Keep `val_acc` as the default for now. Do not switch the global default to `train_acc`.
2. For deeper starters like `big`, test `train_acc` grading again or a mixed train/val score.
3. For `medium_1conv_2linear`, keep validation grading and add an explicit penalty or ban on early stacked sequential dropout.
4. Prefer `medium_1conv_2linear` under `val_acc` when seed stability matters; prefer `big` under `train_acc` when mean accuracy on deeper starts matters.
5. Next experiment should test a dropout-aware scoring rule, not only train-versus-val replacement.
