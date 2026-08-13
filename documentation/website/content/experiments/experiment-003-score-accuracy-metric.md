# Experiment 003: Simulation grading metric

Experiment 002 showed that sequential convolution unlocked medium/small growth, but also that dropout was over-selected. Dropout can raise validation while training stays flat or falls. Simulation graded candidates by validation accuracy, so search could prefer dropout even when learning was weak.

This experiment asks two questions:

1. Does grading simulation candidates by training accuracy reduce dropout bias and improve growth versus grading by validation accuracy?
2. After banning stacked sequential dropout on the same path, which grading mode is better, and is the after-fix grid better overall?

The path ban lives in `AddSeqDropoutLayer.generate_all_actions`: skip pairs that already have Dropout/Dropout2d on the FX path.

Script: `experiments/train_mnist_exp003_score_accuracy_metric.py`

Charts: `documentation/website/scripts/generate_experiment_003_charts.py`

## Two runs

### Before the sequential-dropout path ban

Folder: `experiments/output/train_mnist/runs/exp003_score_accuracy_metric`

Charts: `003-before-*`

Runtime from board metadata: `2026-08-06T18:11:42Z` to `2026-08-06T22:05:37Z`. About `5.4 hours` for `16` completed runs.

### After the sequential-dropout path ban

Folder: `experiments/output/train_mnist/runs/exp003_score_accuracy_metric_after_fix_1`

Charts: `003-after-*`. Compare charts: `003-compare-*`.

Runtime from board metadata: `2026-08-07T16:14:36Z` to `2026-08-07T20:03:33Z`. About `5.1 hours` for `16` completed runs.

## Experiment parameters

Shared by both runs.

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation grading metric | `val_acc`, `train_acc` | Tests train vs val simulation grading |
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

Both grids completed all `16` boards.

---

# Before the fix

Same-path stacked dropout was still allowed.

## Final accuracy by grading mode (before)

![Mean final accuracy before the dropout path ban](/assets/experiments/003-before-final-accuracy-by-score-metric.png)

> [!CAPTION] Figure B1. Before fix. Mean final training and validation accuracy by grading mode × starter. Dots are per-seed finals.

| Grading mode | Starter | Mean final training | Mean final validation | Best final validation | Worst final validation | Mean dropout actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `val_acc` | `big` | `58.69%` | `58.58%` | `88.39%` | `27.57%` | `1.00` |
| `val_acc` | `medium_1conv_2linear` | `52.61%` | `58.27%` | `78.12%` | `25.40%` | `1.00` |
| `train_acc` | `big` | `76.47%` | `76.23%` | `89.57%` | `60.28%` | `0.25` |
| `train_acc` | `medium_1conv_2linear` | `42.55%` | `47.23%` | `77.21%` | `20.57%` | `2.25` |

Low dots are collapses. Under `val_acc`, `big` has weak finals `39.40%` and `27.57%`. Under `train_acc`, `medium_1conv_2linear` has weak finals `20.57%` and `21.33%`. Under `train_acc`, `big` has no collapse: worst final validation is `60.28%`.

Training grading helps `big` on the mean. It hurts `medium_1conv_2linear` because two seeds stack four dropouts.

## Dropout and action composition (before)

![Mean sequential-dropout actions before the path ban](/assets/experiments/003-before-dropout-actions-by-score-metric.png)

> [!CAPTION] Figure B2. Before fix. Mean sequential-dropout actions per completed seed. Dots are individual seeds.

![Executed action counts by type before the path ban](/assets/experiments/003-before-action-composition-by-score-metric.png)

> [!CAPTION] Figure B3. Before fix. Total executed actions by type across all seeds of each grading mode.

| Action type | Count under `val_acc` | Count under `train_acc` | Total |
| --- | ---: | ---: | ---: |
| Add sequential dropout | `8` | `10` | `18` |
| Add residual convolution | `5` | `7` | `12` |
| Add sequential linear | `7` | `3` | `10` |
| Add sequential convolution | `3` | `2` | `5` |
| Add residual linear | `1` | `0` | `1` |
| Delete layer | `0` | `1` | `1` |

### Why does `train_acc` show more dropout than `val_acc`?

This looks backwards if the Exp 002 story is “validation grading over-selects dropout.” The counts are not wrong. The reason is stacking on collapsed seeds, not “validation avoids dropout at the first pick.”

- First-action dropout count: `val_acc` `4/8` seeds, `train_acc` `3/8` seeds. Validation grading starts with dropout at least as often.
- Under `train_acc` × `medium_1conv_2linear`, seeds `101` and `102` then stack four dropouts each on the same edge between the two linear layers. Those eight stacked actions inflate the train-grading total.
- Under `val_acc`, weak seeds usually stop at one or two dropouts. Strong validation seeds often start with sequential linear, then residual convolution.

So validation grading has fewer total dropout actions because it does not enter the four-stack spiral as hard. It does not mean validation search stopped liking dropout.

### Why more sequential linear under `val_acc`, more residual convolution under `train_acc`?

This is also measured, not a chart bug.

- Under `val_acc`, first actions on strong seeds are often sequential linear (`big` seeds `100`, `102`; `medium` seed `100`). Residual convolution comes later. That is why sequential linear totals `7` and residual convolution totals only `5`.
- Under `train_acc`, strong `big` seeds start with residual convolution more often (`101`, `103`). Residual convolution totals `7`.
- Exp 002 also used validation grading and still executed dropout more than residual convolution overall. Validation grading does not always mean “residual convolution first.”

After the path ban, validation grading does look more like the expected residual-convolution-heavy pattern.

## Accuracy gain after architecture actions (before)

For each executed action, compare end-of-generation training and validation accuracy with the end of the next generation.

![Training and validation change by action type before the path ban](/assets/experiments/003-before-action-types.png)

> [!CAPTION] Figure B4. Before fix. Mean training and validation change by action type. This chart pools every executed action from both grading modes. It is not a simple average of the two panels in Figure B5; rare actions keep their own counts. Bars are means. Dots are individual actions.

| Action type | n | Mean training change | Mean validation change | Non-positive training | Non-positive validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Add residual convolution | `12` | `+38.21 percentage points` | `+37.34 percentage points` | `0` | `0` |
| Add residual linear | `1` | `+7.61 percentage points` | `+12.10 percentage points` | `0` | `0` |
| Add sequential convolution | `5` | `+6.40 percentage points` | `+7.48 percentage points` | `0` | `0` |
| Add sequential linear | `10` | `+6.24 percentage points` | `+5.26 percentage points` | `0` | `1` |
| Delete layer | `1` | `+3.79 percentage points` | `+12.94 percentage points` | `0` | `0` |
| Add sequential dropout | `18` | `-0.46 percentage points` | `+2.18 percentage points` | `10` | `6` |

![Action-type accuracy change by grading mode before the path ban](/assets/experiments/003-before-action-types-by-score-metric.png)

> [!CAPTION] Figure B5. Before fix. Same recovery window as Figure B4, split by grading mode.

| Grading mode | Action type | n | Mean training change | Mean validation change |
| --- | --- | ---: | ---: | ---: |
| `val_acc` | Add residual convolution | `5` | `+43.60` | `+41.12` |
| `val_acc` | Add sequential dropout | `8` | `-0.72` | `+3.00` |
| `train_acc` | Add residual convolution | `7` | `+34.35` | `+34.65` |
| `train_acc` | Add sequential dropout | `10` | `-0.26` | `+1.53` |

Values are percentage points. Residual convolution has the largest gain under both modes. Dropout is the weak learning action.

## Training histories (before)

![Training curves before the path ban](/assets/experiments/003-before-training-curves.png)

> [!CAPTION] Figure B6. Before fix. Training curves for each grading mode × starter. Line color marks the seed.

### `val_acc` + `big`

Two strong seeds climb without early dropout. Two weak seeds start with dropout and stay low.

### `train_acc` + `big`

All four seeds stay useful. This is why train grading wins the before-fix pooled mean.

### `val_acc` + `medium_1conv_2linear`

Mixed. One hard collapse after early stacked dropout. One clear mid-run training crash after a later dropout.

### `train_acc` + `medium_1conv_2linear`

Seeds `101` and `102` each insert four sequential dropouts between the same two linear modules and collapse near `21%` validation. That is the same-path stacking bug.

## Which grading wins before the fix?

![Overall grade val vs grade train before the path ban](/assets/experiments/003-before-grading-overall-final-validation.png)

> [!CAPTION] Figure B7. Before fix. Pooled mean final validation for each grading mode. Dots are all seeds of that mode.

![Grade val vs grade train by starter before the path ban](/assets/experiments/003-before-grading-by-model-final-validation.png)

> [!CAPTION] Figure B8. Before fix. Same comparison split by starter.

| Starter | Grade `val_acc` | Grade `train_acc` | Better before the fix |
| --- | ---: | ---: | --- |
| `big` | `58.58%` | `76.23%` | `train_acc` |
| `medium_1conv_2linear` | `58.27%` | `47.23%` | `val_acc` |
| Both starters pooled | `58.42%` | `61.73%` | `train_acc` on the mean only |

## Before-fix conclusions

1. Pooled answer before the fix: `train_acc` looks slightly better (`61.73%` vs `58.42%`), but only because `big` is strong. Medium is worse under train grading.
2. More total dropout under `train_acc` comes from four-stack collapses, not from fewer first-action dropouts under `val_acc`.
3. Residual convolution always helps. Sequential dropout often helps validation while training stays flat or falls.
4. Same-path stacked dropout must be banned.

---

# After the fix

Same-path stacked dropout is banned.

## Final accuracy by grading mode (after)

![Mean final accuracy after the dropout path ban](/assets/experiments/003-after-final-accuracy-by-score-metric.png)

> [!CAPTION] Figure A1. After fix. Mean final training and validation accuracy by grading mode × starter. Dots are per-seed finals.

| Grading mode | Starter | Mean final training | Mean final validation | Best final validation | Worst final validation | Mean dropout actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `val_acc` | `big` | `70.32%` | `71.27%` | `84.40%` | `46.03%` | `0.00` |
| `val_acc` | `medium_1conv_2linear` | `52.26%` | `66.54%` | `72.42%` | `61.57%` | `0.75` |
| `train_acc` | `big` | `65.56%` | `65.73%` | `83.40%` | `32.46%` | `0.75` |
| `train_acc` | `medium_1conv_2linear` | `44.03%` | `50.08%` | `70.83%` | `23.59%` | `0.75` |

## Dropout and action composition (after)

![Mean sequential-dropout actions after the path ban](/assets/experiments/003-after-dropout-actions-by-score-metric.png)

> [!CAPTION] Figure A2. After fix. Mean sequential-dropout actions per completed seed. Dots are individual seeds.

![Executed action counts by type after the path ban](/assets/experiments/003-after-action-composition-by-score-metric.png)

> [!CAPTION] Figure A3. After fix. Total executed actions by type across all seeds of each grading mode.

| Action type | Count under `val_acc` | Count under `train_acc` | Total |
| --- | ---: | ---: | ---: |
| Add residual convolution | `12` | `9` | `21` |
| Add sequential dropout | `3` | `6` | `9` |
| Add sequential linear | `2` | `5` | `7` |
| Add sequential convolution | `4` | `2` | `6` |
| Delete layer | `2` | `3` | `5` |

### Why does `val_acc` still use less dropout after the fix?

Now the pattern matches the selection behavior more cleanly:

- First-action dropout: `val_acc` `2/8`, `train_acc` `4/8`.
- Total dropout: `val_acc` `3`, `train_acc` `6`.
- Residual convolution: `val_acc` `12`, `train_acc` `9`.

So after the ban, validation grading really does pick residual convolution more and dropout less. The earlier before-fix “less dropout under val” effect was mostly stacking inflation under train grading. After the ban, the cleaner reading is: validation grading prefers growth actions more often in this grid.

No run repeats dropout between the same two modules. The only multi-dropout run places one dropout on the conv path and one on the linear path (`train_acc` × `big` seed `102`). Different edges are still allowed.

## Accuracy gain after architecture actions (after)

![Training and validation change by action type after the path ban](/assets/experiments/003-after-action-types.png)

> [!CAPTION] Figure A4. After fix. Mean training and validation change by action type. Pooled across both grading modes. Bars are means. Dots are individual actions.

| Action type | n | Mean training change | Mean validation change | Non-positive training | Non-positive validation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Add residual convolution | `21` | `+22.69 percentage points` | `+25.23 percentage points` | `0` | `1` |
| Add sequential convolution | `6` | `+7.44 percentage points` | `+7.63 percentage points` | `0` | `0` |
| Add sequential dropout | `9` | `+3.55 percentage points` | `+6.09 percentage points` | `1` | `0` |
| Add sequential linear | `7` | `+3.38 percentage points` | `+3.85 percentage points` | `1` | `0` |
| Delete layer | `5` | `-0.28 percentage points` | `-1.01 percentage points` | `3` | `2` |

![Action-type accuracy change by grading mode after the path ban](/assets/experiments/003-after-action-types-by-score-metric.png)

> [!CAPTION] Figure A5. After fix. Same recovery window as Figure A4, split by grading mode.

Residual convolution remains the high-gain action. Delete layer is the weak action on average.

## Training histories (after)

![Training curves after the path ban](/assets/experiments/003-after-training-curves.png)

> [!CAPTION] Figure A6. After fix. Training curves for each grading mode × starter. Line color marks the seed.

### `val_acc` + `big`

No dropout in this cell. Residual convolution dominates. Three seeds climb high. One stays moderate.

### `train_acc` + `big`

Three strong seeds. One collapse after two dropouts on different edges.

### `val_acc` + `medium_1conv_2linear`

All four seeds finish above `61%` validation. No same-path stacking.

### `train_acc` + `medium_1conv_2linear`

Four-stack collapses are gone. One early single dropout can still leave a weak seed.

## Chosen architectures under validation grading after the fix

Because the after-fix comparison favors `val_acc`, the graphs below show the architectures we would keep: validation grading after the path ban. Starts and finals are simplified FX graphs from the after-fix boards.

![big starter under after-fix validation grading](/assets/experiments/exp003-graphs/start-big-valacc.png)

> [!CAPTION] Figure A7. After fix, `val_acc`, `big` starter. `420` parameters. Seed `101` start graph.

![Best big final under after-fix validation grading](/assets/experiments/exp003-graphs/final-big-seed101-val84.png)

> [!CAPTION] Figure A8. After fix, `val_acc`, best `big` final: seed `101`, validation `84.40%`. Actions: two residual convolutions. Parameters `420` → `1160`.

![Moderate big final under after-fix validation grading](/assets/experiments/exp003-graphs/final-big-seed100-val46.png)

> [!CAPTION] Figure A9. After fix, `val_acc`, moderate `big` final: seed `100`, validation `46.03%`. Actions: three residual convolutions. Parameters `420` → `864`.

![medium_1conv_2linear starter under after-fix validation grading](/assets/experiments/exp003-graphs/start-medium_1conv_2linear-valacc.png)

> [!CAPTION] Figure A10. After fix, `val_acc`, `medium_1conv_2linear` starter. `276` parameters. Seed `102` start graph.

![Best medium final under after-fix validation grading](/assets/experiments/exp003-graphs/final-medium_1conv_2linear-seed103-val72.png)

> [!CAPTION] Figure A11. After fix, `val_acc`, best `medium_1conv_2linear` final: seed `103`, validation `72.42%`. Actions: sequential dropout, then residual convolution. Parameters `276` → `868`.

![Strong medium final without early dropout](/assets/experiments/exp003-graphs/final-medium_1conv_2linear-seed102-val70.png)

> [!CAPTION] Figure A12. After fix, `val_acc`, strong `medium_1conv_2linear` final: seed `102`, validation `70.16%`. Actions: residual convolution, then sequential linear. Parameters `276` → `1140`. No dropout.

## Which grading wins after the fix?

![Overall grade val vs grade train after the path ban](/assets/experiments/003-after-grading-overall-final-validation.png)

> [!CAPTION] Figure A13. After fix. Pooled mean final validation for each grading mode. Dots are all seeds of that mode.

![Grade val vs grade train by starter after the path ban](/assets/experiments/003-after-grading-by-model-final-validation.png)

> [!CAPTION] Figure A14. After fix. Same comparison split by starter.

| Starter | Grade `val_acc` | Grade `train_acc` | Better after the fix |
| --- | ---: | ---: | --- |
| `big` | `71.27%` | `65.73%` | `val_acc` |
| `medium_1conv_2linear` | `66.54%` | `50.08%` | `val_acc` |
| Both starters pooled | `68.90%` | `57.91%` | `val_acc` |

Answer after the fix: validation grading is better overall and on both starters.

## After-fix conclusions

1. Pooled answer after the fix: `val_acc` is clearly better (`68.90%` vs `57.91%`).
2. Same-path stacking is gone.
3. Validation grading becomes more residual-convolution heavy and uses less dropout.
4. Early dropout on different edges can still hurt one seed under train grading.
5. The chosen default path is after-fix `val_acc`, with residual-convolution growth on both starters.

---

# Compare before and after

![Mean final validation before vs after by cell](/assets/experiments/003-compare-final-validation-by-score-metric.png)

> [!CAPTION] Figure C1. Mean final validation by grading mode × starter, before vs after. Dots are seeds.

![Mean dropout actions before vs after by cell](/assets/experiments/003-compare-dropout-actions-by-score-metric.png)

> [!CAPTION] Figure C2. Mean sequential-dropout actions by grading mode × starter, before vs after.

| Cell | Before mean final val | After mean final val | Before mean dropout | After mean dropout |
| --- | ---: | ---: | ---: | ---: |
| `val_acc` × `big` | `58.58%` | `71.27%` | `1.00` | `0.00` |
| `val_acc` × `medium_1conv_2linear` | `58.27%` | `66.54%` | `1.00` | `0.75` |
| `train_acc` × `big` | `76.23%` | `65.73%` | `0.25` | `0.75` |
| `train_acc` × `medium_1conv_2linear` | `47.23%` | `50.08%` | `2.25` | `0.75` |

| Quantity | Before | After |
| --- | ---: | ---: |
| Overall mean final validation | `60.08%` | `63.41%` |
| Grade `val_acc` mean final validation | `58.42%` | `68.90%` |
| Grade `train_acc` mean final validation | `61.73%` | `57.91%` |
| Total sequential dropout actions | `18` | `9` |
| Total residual convolution actions | `12` | `21` |
| Same-path stacked dropout | yes | no |

### What improved

- After the fix is better overall (`60.08%` → `63.41%`).
- Same-path stacked dropout disappears.
- Total dropout falls by half.
- Residual convolution nearly doubles.
- Validation grading becomes clearly better and more stable on both starters.

### What did not fully improve

- Train grading on `big` gets worse on the mean because one seed collapses with dropout on two different edges.
- A single early dropout can still leave a weak medium seed under train grading.

## Answers to the two experiment questions

![Overall mean final validation before vs after](/assets/experiments/003-compare-overall-before-after.png)

> [!CAPTION] Figure Q1. Mean final validation over all `16` seeds in each phase. Dots are individual seeds.

| Phase | Mean final validation | Mean dropout actions |
| --- | ---: | ---: |
| Before fix | `60.08%` | `1.12` |
| After fix | `63.41%` | `0.56` |

![Grade val vs grade train by phase](/assets/experiments/003-compare-grading-overall-by-phase.png)

> [!CAPTION] Figure Q2. Mean final validation by grading mode. Each bar pools both starters. Dots are seeds.

| Phase | Grade `val_acc` mean final val | Grade `train_acc` mean final val | Winner |
| --- | ---: | ---: | --- |
| Before fix | `58.42%` | `61.73%` | `train_acc` by `3.31` percentage points |
| After fix | `68.90%` | `57.91%` | `val_acc` by `10.99` percentage points |

1. After the path ban, the after-fix grid is better overall than the before-fix grid.
2. Before the fix, train grading only looked better on the pooled mean because of strong `big` seeds.
3. After the fix, validation grading is clearly better overall and on both starters.
4. Keep the path ban, and keep `val_acc` as the default simulation grading metric.

## Overall conclusions

1. After the fix is better than before the fix on the pooled mean (`60.08%` → `63.41%`).
2. Before the fix, train grading only looked better because of strong `big` seeds. Medium was worse.
3. After the fix, validation grading is the clear winner overall and on both starters (`68.90%` vs `57.91%`).
4. The path ban is necessary. It removes same-path stacking. It does not remove every bad early dropout.
5. Residual convolution remains the high-value growth action.
6. Keep `val_acc` as the default simulation grading metric, with the after-fix dropout path ban committed.

## Next steps

1. Keep the same-path dropout ban in `AddSeqDropoutLayer.generate_all_actions`.
2. Keep `val_acc` as the default. Do not switch the global default to `train_acc`.
3. Prefer residual convolution early.
4. Later learning-rate and simulation-scheduler experiments should use after-fix `val_acc` as the baseline package.
5. If another dropout experiment is needed, study early multi-edge dropout under train grading, not a global “only one dropout in the whole model” ban.
6. Commit the `003-before-*`, `003-after-*`, `003-compare-*`, and `exp003-graphs` assets plus the JSON snapshot before deleting raw output folders.
