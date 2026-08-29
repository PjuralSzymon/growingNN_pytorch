# Experiment 008: CIFAR-10 adaptive meta-parameter search

GrowingNN on sequential CIFAR-10 is in the high 70s, not the 90s. After `30` / `50` adaptive-search trials, the best peak is `79.12%` on the `deep` starter with `400` epochs and architecture growth. Starter width and training length dominate. Search algorithm and learning-rate details are secondary.

This page is a live report. Trial `31` is still running. Refresh `adaptive_search.md`, then re-run the chart script.

Script: `experiments/train_cifar10_exp008_initial_package.py`

Search: `experiments/adaptive_metaparameter_search.py`

Charts: `documentation/website/scripts/generate_experiment_008_charts.py`

Folder: `experiments/output/train_cifar10/runs/exp008_cifar10_initial_package`

Snapshot: `documentation/website/data/experiments/experiment-008-cifar10-initial-package.json`

The snapshot and PNG files are untracked documentation artifacts. Commit them before deleting raw `experiments/output/` data.

CIFAR-10 has no held-out val split here. `Cifar10Data` uses the official test set as `val_loader`. The search stores the same number as `val_acc` and `test_acc`. That number is peak accuracy (`max(val_acc)`), not the last epoch.

## Where we are

The MNIST keep-cell from Experiments 000 to 005 is not the CIFAR winner. The current best cell is `deep` + `20` epochs × `20` generations + `sequential_halving_beam` + `composed_step` + `lr_alpha=0.03` + `120` s + sim epochs `15` + set `1000` + `slope_3deg`.

Pieces that already work on this CIFAR stem:

- Neuron-resize is on. Across the `30` boards: Add Neurons `117`, Delete Neurons `87`, Add Res Conv `63`, Add Seq Linear `58`, dropout `53`, Delete Layer `40`, Add Res Linear `30`.
- Residual conv growth is used, not only linear inserts.
- The `3°` slope gate still opens. It is not a no-search control.
- Flatten-width repair is in product code, so Add/DelNeurons can run after MaxPool flatten.

What this run cannot yet answer: whether growth beats a fixed `deep` net, whether `79%` repeats on extra seeds, and how far a residual CIFAR stem can go. A published ResNet-18 on CIFAR-10 is typically near `93%`. This starter is a sequential `MinimalCifarNet`.

## Experiment parameters

| Group | Values |
| --- | --- |
| `starter` | `narrow` 4/32 (`33390` params), `base` 8/38 (`79060`), `mid` 12/45 (`140389`), `deep` 16/48 (`199914`) |
| `epochs` × `generations` | `5`/`10`/`20` × `10`/`20` |
| `simulation_alg` | `montecarlo`, `greedy`, `sequential_halving_beam`, `ugape_deepen`, `best_first` |
| `lr_schedule` | `composed_exponential`, `composed_step`, `composed_cosine` |
| `lr_alpha` | `0.001`, `0.01`, `0.03` |
| `simulation_time` | `60`, `120`, `240` s |
| `simulation_epochs` | `10`, `15`, `20` |
| `simulation_set_size` | `100`, `500`, `1000` |
| `simulation_scheduler` | `always`, `slope_2deg`, `slope_3deg` |

| Fixed | Value |
| --- | --- |
| Dataset | CIFAR-10, official test used as val |
| Pool / budget | `87480` combos, `50` trials, `5` warm-up |
| Seeds | one per trial, trial `k` uses `100 + k - 1` |
| Batch / loss | `64`, `CrossEntropyLoss` |
| Neuron-resize flags | on (`RunningConfig` defaults) |
| Simulation-set generator | `protected` (Experiment 007 unfinished) |
| Augmentation | RandomCrop pad `4` + horizontal flip |

`always` runs architecture search every generation. `slope_Xdeg` is `SlopeEstimationSimulationScheduler`: search only when the accuracy curve is flatter than `X` degrees.

## Result timeline

Progress: `30` / `50` (`60.0%`). Unevaluated remaining: `87449` / `87480`.

Board timestamps (`experimentStartedOn`): first `2026-08-25T19:55:20Z`, last completed `2026-08-29T08:36:16Z`. Summed wall time across the `30` boards is `86.1` hours.

| Field | Value |
| --- | --- |
| Best peak `val_acc` | `79.12%` (trial `21`, seed `120`) |
| Best final val / train | `78.67%` / `76.70%` |
| Best params | `199914` → `365490` |
| Best sims / actions | `18` / `18` |
| Next pending combo | `deep`, `20`/`20`, `ugape_deepen`, `composed_step`, `lr_alpha=0.01`, `60` s, sim epochs `15`, set `100`, `slope_3deg` |

This search restarted after neuron-resize was turned on. It is not the older 15-trial board that peaked at `70.94%`.

## Why this experiment

Copying the MNIST keep-cell onto CIFAR cannot tell which knobs are wrong. `AdaptiveMetaParameterSearch` samples the product of the groups, then raises the probability of values with high mean peak validation accuracy.

## Measurements and charts

### Which trials are strongest?

One seed per combo. Color is starter width.

![Peak validation accuracy by search trial](/assets/experiments/008-trial-val-acc.png)

> [!CAPTION] Figure 1. Peak validation accuracy (%) for each scored trial. Color is starter width. The y label also shows the simulation scheduler.

The top three are all `deep` and `400` epochs except t13 (`deep`, `200` epochs, `75.62%`): t21 `79.12%`, t11 `77.37%`, t13 `75.62%`. Every `narrow` trial is at or below `61.42%`.

### Does starter width dominate?

Mean peak by starter, with one marker per trial.

![Peak validation accuracy by starter](/assets/experiments/008-starter-peak-val.png)

> [!CAPTION] Figure 2. Bars are mean peak validation accuracy (%) by starter. Gray points are the individual trials.

`deep` `n=8` mean `74.84%` (`70.58` to `79.12`). `mid` `n=12` mean `72.13%`. `base` `n=5` mean `66.08%`. `narrow` `n=5` mean `51.87%`. Drop `narrow` from later CIFAR cells.

### Does training length dominate the rest?

Total epochs are `epochs × generations`. Markers are jittered on x so overlapping trials stay visible.

![Peak validation accuracy versus training length](/assets/experiments/008-epochs-vs-val.png)

> [!CAPTION] Figure 3. Peak validation accuracy (%) versus total training epochs. Color is starter. Marker is the simulation scheduler: circle `always`, triangle `2°`, square `3°`. Horizontal jitter is only for overlap.

`400` epochs `n=16` mean `72.99%`. `200` `n=7` mean `68.77%`. `100` `n=6` mean `59.67%`. The one `50`-epoch trial is `46.82%` and is also `narrow`. Length and width are mixed in the grades. Both still beat algorithm identity in this sample.

### Did the peak survive?

The search records `max(val_acc)`. A late collapse can still get a high search score.

![Peak versus final validation accuracy](/assets/experiments/008-peak-vs-final-val.png)

> [!CAPTION] Figure 4. Peak versus final validation accuracy (%). Points on the diagonal kept their peak. Color is starter.

Three collapses (peak minus final greater than `5` percentage points): t1 `10.0` (`narrow`, params `33390` → `5630`), t16 `23.6` (`mid`, `140389` → `43529`), t28 `7.3` (`mid`, `140389` → `56728`). Trial 21 stayed near the diagonal (`79.12` peak, `78.67` final).

### What does a good CIFAR run look like?

Trial 21 is the current winner: `deep`, `400` epochs, `sequential_halving_beam`, `composed_step`, `lr_alpha=0.03`, `slope_3deg`.

![Best trial train and validation accuracy](/assets/experiments/008-best-trial-curves.png)

> [!CAPTION] Figure 5. Trial 21 training and validation accuracy (%) over `400` epochs. Validation is the official CIFAR-10 test split.

Train sits below val because train uses RandomCrop and flip. The curve climbs through the first ~`150` epochs, then jitters in the high `70s`. It does not show the t16 shrink-and-collapse shape.

### What do the axis grades say?

Unused values start at grade `0.5`. After a trial, `raw` is the mean peak `val_acc` of scored combos with that value. Then `grade = 0.7 * old + 0.3 * raw`. Grades mix the named axis with the rest of those combos.

![Per-axis search grades](/assets/experiments/008-axis-grades.png)

> [!CAPTION] Figure 6. Current EMA grades by group. The gray line is the unused-value start `0.5`.

Highest grades: `deep` `0.753`, epochs `20` `0.724`, generations `20` `0.711`, `composed_cosine` `0.716`, `lr_alpha=0.03` `0.717`, `slope_3deg` `0.697`. `greedy` is `0.444` from one `narrow` trial. Do not read that as “greedy is broken.” The live winner uses `sequential_halving_beam` and `composed_step`, which are not the highest mean grades.

### Did the networks grow?

![Start and final parameter counts by trial](/assets/experiments/008-param-growth.png)

> [!CAPTION] Figure 7. Start (gray) and final (blue) parameter counts for each scored trial.

Most trials grow. The three collapse trials shrink. Trial 21 grew `199914` → `365490`.

### What actions ran?

![Executed action mix by trial](/assets/experiments/008-action-composition.png)

> [!CAPTION] Figure 8. Executed action counts by short label for each scored trial.

Add/Delete Neurons are the two most common live actions. This is not the Experiment 006 MNIST grid, where those actions almost never ran. Do not wait on Experiment 006 to decide neuron-resize for CIFAR.

## Grouped final results

Mean peak validation accuracy by axis (confounded with the rest of each combo):

| Axis | Strongest in this sample | Weakest in this sample |
| --- | --- | --- |
| starter | `deep` `74.84%` (`n=8`) | `narrow` `51.87%` (`n=5`) |
| total epochs | `400` `72.99%` (`n=16`) | `50` `46.82%` (`n=1`) |
| `lr_alpha` | `0.03` `71.57%` (`n=15`) | `0.001` `61.56%` (`n=8`) |
| LR schedule | cosine `71.60%` (`n=12`) | step `65.83%` (`n=10`) |
| scheduler | `slope_3deg` `69.75%` (`n=20`) | `slope_2deg` `61.82%` (`n=3`) |
| sim set | `500` `72.33%` (`n=5`) | `100` `64.80%` (`n=12`) |

Top five trials:

| Trial | Seed | Starter | Total ep | Alg | LR | Peak val (%) | Final val (%) | Params |
| --- | ---: | --- | ---: | --- | --- | ---: | ---: | --- |
| 21 | 120 | deep | 400 | sequential_halving_beam | step `0.03` | 79.12 | 78.67 | 199914 → 365490 |
| 11 | 110 | deep | 400 | ugape_deepen | cosine `0.03` | 77.37 | 77.32 | 199914 → 251948 |
| 13 | 112 | deep | 200 | montecarlo | cosine `0.01` | 75.62 | 74.59 | 199914 → 296744 |
| 17 | 116 | deep | 400 | sequential_halving_beam | exponential `0.001` | 74.66 | 73.22 | 199914 → 260199 |
| 19 | 118 | mid | 400 | ugape_deepen | cosine `0.03` | 74.44 | 74.29 | 140389 → 172185 |

Trial 21 actions, in order: res conv ×2, add neurons, seq linear ×2, dropout ×2, seq linear, res linear ×2, dropout ×2, delete layer, del neurons, seq linear, del neurons, res linear, add neurons.

## Training-history analysis

Curves have different lengths because total epochs are `50`, `100`, `200`, or `400`.

![Validation accuracy curves by starter](/assets/experiments/008-validation-curves.png)

> [!CAPTION] Figure 9. Validation accuracy (%) over epochs for every scored trial. Color is starter. The CIFAR-10 test split is logged as `val_acc`.

`deep` and `mid` separate from `narrow` within about `50` epochs. `narrow` stays near `45` to `60%` even on long runs. The sharp drops are the shrink trials (t1, t16, t28), not the slope-`3°` gate itself.

## Limitations and seed effects

- `30` trials cannot cover `87480` combos. One seed per combo. Gaps of `1` to `2` percentage points can be seed luck.
- Peak `val_acc` is the official CIFAR-10 test split. A collapse after the peak still scores the peak.
- Axis grades confound the named value with the rest of the combo. `greedy` has one `narrow` trial.
- There is no `never`-search control in this pool, so growth vs fixed architecture is not measured.
- Sequential `MinimalCifarNet` is not a residual CIFAR baseline.
- Simulation-set generators stay at `protected` because Experiment 007 is unfinished.

## Conclusions

These statements use the `30` scored trials only.

1. Current CIFAR status is the high `70s` on a sequential two-conv stem. Best peak `79.12%` (trial 21). That is progress from the old 15-trial report (`70.94%`), not a CIFAR-10 SOTA result.
2. Starter width and total epochs are the two knobs that move accuracy. `narrow` is a dead cell. Prefer `deep` (or `mid`) and `400` epochs.
3. Architecture actions that were already implemented are being used: neuron-resize, residual conv, sequential linear, dropout, delete. The algorithm is not missing those operators on this stem.
4. The MNIST keep package is not the CIFAR keep package. Do not copy `composed_exponential` + `lr_alpha=0.01` + `10`×`10` as the CIFAR default.
5. Do not lock algorithm or LR yet. Cosine has the higher mean. Step has the winner. `greedy` is confounded.

## Next experiments

1. Finish the remaining `20` trials. Re-run this page from `adaptive_search.json`.
2. Repeat the winning combo on seeds `121` and `122`.
3. Add a `never`-search control on the same `deep` + `400`-epoch + `composed_step` + `0.03` cell. That is the missing “does growth help?” measurement.
4. Keep neuron-resize on. Experiment 006 is inconclusive and should not gate CIFAR.
5. After Experiment 007, swap `protected` for the winning simulation-set generator on the CIFAR winner, not on MNIST `big`.
6. Only after the control and seed repeats: try a residual CIFAR starter. The sequential stem is the likely ceiling for this `79%`, not a missing search algorithm.
