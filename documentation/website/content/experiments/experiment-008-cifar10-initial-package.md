# Experiment 008: CIFAR-10 adaptive meta-parameter search

MNIST keep-settings from Experiments 000 to 005 are likely wrong for CIFAR-10. This experiment does not copy one MNIST cell. It searches a Cartesian pool of meta-parameter groups with `AdaptiveMetaParameterSearch` (issue 73).

This page is a live report of `15` / `50` scored trials (`30.0%`). Trial `16` is still running. Conclusions below are provisional. Refresh `adaptive_search.md` while the search continues, then re-run the chart script.

Script: `experiments/train_cifar10_exp008_initial_package.py`

Search module: `experiments/adaptive_metaparameter_search.py`

Charts: `documentation/website/scripts/generate_experiment_008_charts.py`

Folder: `experiments/output/train_cifar10/runs/exp008_cifar10_initial_package`

Live status: `experiments/output/train_cifar10/runs/exp008_cifar10_initial_package/adaptive_search.md`

Snapshot: `documentation/website/data/experiments/experiment-008-cifar10-initial-package.json`

The snapshot and PNG files are untracked documentation artifacts. Commit them before deleting raw `experiments/output/` data.

CIFAR-10 has no held-out val split in this driver. `Cifar10Data` uses the official test set as `val_loader`. The search therefore records the same number as `val_acc` and `test_acc`. That number is peak accuracy over the run (`max(val_acc)`), not the last epoch.

## Experiment parameters

| Group | Values | Purpose |
| --- | --- | --- |
| `starter` | `narrow` 4/32 (`33390` params), `base` 8/38 (`79060`), `mid` 12/45 (`140389`), `deep` 16/48 (`199914`) | Sequential `MinimalCifarNet` width |
| `epochs` | `5`, `10`, `20` | Epochs per generation |
| `generations` | `10`, `20` | Architecture-decision cycles |
| `simulation_alg` | `montecarlo`, `greedy`, `sequential_halving_beam`, `ugape_deepen`, `best_first` | How candidates are searched. No `random` |
| `lr_schedule` | `composed_exponential`, `composed_step`, `composed_cosine` | Exp 004 keep trio |
| `lr_alpha` | `0.001`, `0.01`, `0.03` | Peak / target learning rate |
| `simulation_time` | `60`, `120`, `240` s | Wall-time budget inside one search call |
| `simulation_epochs` | `10`, `15`, `20` | Scoring GD steps inside simulation |
| `simulation_set_size` | `100`, `500`, `1000` | Samples used by simulation scoring |
| `simulation_scheduler` | `always`, `slope_2deg`, `slope_3deg` | When architecture search may run |

`simulation_scheduler` is not a learning-rate setting and is not the search algorithm. `always` runs architecture search every generation (`AlwaysSimulationScheduler`). `slope_Xdeg` is `SlopeEstimationSimulationScheduler`: if the accuracy curve is flatter than `X` degrees, treat training as stuck and allow search; if the curve is still climbing, skip search and keep SGD.

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | CIFAR-10 | Harder image task than MNIST |
| Pool size | `87480` | Full Cartesian product of the ten groups |
| Search budget | `50` trials | `max_iters`. Not the full pool |
| Scored so far | `15` | This refresh |
| Warm-up trials | `5` | Uniform samples (`n_init`) |
| Softmax temperature `tau` | `0.15` | `P_a = Softmax(grade[a] / tau)` |
| Grade EMA `beta` | `0.3` | `grade = (1-beta)*old + beta*mean(val)` |
| Seeds | one per trial, start `100` | Trial `k` uses seed `100 + k - 1` |
| Batch size | `64` | Training samples per batch |
| `target_accuracy` | `0.99` | `AccuracyStopper` |
| Accuracy metric | `val_acc` | Simulation grading |
| Score weight accuracy | `1.0` | `score_weight_acc` |
| Score weight parameter count | `0.1` | `score_weight_countw` |
| Recovery warmup | logistic, `warmup_iterations=10`, `k=10` | After an architecture action |
| Minimum LR floor | `0.001` | `MIN_LEARNING_RATE` |
| Exponential gamma | `0.98` | Base decay for `composed_exponential` |
| Step gamma / step size | `0.5`, `total_epochs // 3` | `composed_step` |
| Cosine `eta_min` | `0.001` | `composed_cosine` |
| Neuron-resize flags | off | Experiment 006 is unfinished |
| Same-path dropout ban | on | Experiment 003 product rule |
| Simulation-set generator | `protected` | Experiment 007 is unfinished |
| Look-ahead | depth `2`, beam `3` | Beam / sequential-halving constants |
| Residual-to-linear pool | average | Used if growth inserts a residual conv before a linear |
| Augmentation | RandomCrop pad `4` + horizontal flip | CIFAR train transform in `Cifar10Data` |
| Loss | `CrossEntropyLoss` | Training criterion |
| Validation loader | official CIFAR-10 test split | Recorded `val_acc` is also `test_acc` |

Each trial does three steps in `evaluate_combo`: create the model (`train_cifar10._build_model` then FX-trace), train (`train_generations`), save (`experiments_common._save_artifacts`).

Run path:

```text
exp008_cifar10_initial_package/<combo_folder>/seed_<seed>/
```

Watch the live tables:

```text
experiments/output/train_cifar10/runs/exp008_cifar10_initial_package/adaptive_search.md
```

Rebuild charts from the current search file:

```text
python documentation/website/scripts/generate_experiment_008_charts.py
```

## Research questions

Main question: which CIFAR-10 meta-parameter combo should replace a copied MNIST keep-cell?

Supporting checks:

1. Which starter width (`narrow`, `base`, `mid`, `deep`) has a high grade after the scored trials?
2. Does architecture search need `always`, or does a `2°` / `3°` slope gate still fire on CIFAR?
3. Do generation length and epoch count change the grade more than the simulation algorithm?
4. Is `composed_exponential` still the LR keep-set, or do `composed_step` / `composed_cosine` win on CIFAR?

## Result timeline

Progress: `15` / `50` scored trials (`30.0%`). Unevaluated remaining: `87464` / `87480`.

Board timestamps (`experimentStartedOn`): first trial `2026-08-21T14:40:08Z`. Trial 15 started `2026-08-22T07:40:37Z`. Summed wall time across the 15 completed boards is `63036` s (`17.5` hours).

| Field | Value |
| --- | --- |
| Best combo | `deep`, `10` epochs, `10` generations, `sequential_halving_beam`, `composed_exponential`, `lr_alpha=0.03`, `60` s, `20` sim epochs, set `100`, `always` |
| Best peak `val_acc` | `70.94%` (trial 1, seed `100`) |
| Best `test_acc` | same number (CIFAR test split) |
| Next pending combo | `base`, `10` epochs, `20` generations, `greedy`, `composed_step`, `lr_alpha=0.001`, `240` s, `20` sim epochs, set `1000`, `slope_2deg` (seed `115`) |

Never sampled yet: `composed_cosine`, `simulation_epochs=10`. Those grades stay at `0.5`.

## Why this experiment

Experiments 000 to 005 were MNIST. Copying `sequential_halving_beam` + `composed_exponential` + `val_acc` + 3° slope onto CIFAR cannot tell which knobs are wrong. Adaptive search samples the product of the groups, then raises the probability of values that have high mean peak validation accuracy.

## Measurements and charts

### Which trials are strongest so far?

The search ranks combos by peak validation accuracy. One seed per combo.

![Peak validation accuracy by search trial](/assets/experiments/008-trial-val-acc.png)

> [!CAPTION] Figure 1. Peak validation accuracy (%) for each scored trial. Color is starter width. The y label also shows the simulation scheduler.

The top three are still `70.94%` (t1 `deep` / `always`), `70.91%` (t7 `deep` / `always`), and `70.84%` (t10 `mid` / `always`). The two new trials sit lower: t14 `mid` / `3°` at `63.79%`, t15 `narrow` / `2°` at `55.38%`. Every `narrow` trial is below `58%`. Starter width still separates the ranking more than the scheduler label.

### Did the peak survive until the last epoch?

The search records `max(val_acc)`. A late collapse can still get a high search score.

![Peak versus final validation accuracy](/assets/experiments/008-peak-vs-final-val.png)

> [!CAPTION] Figure 2. Peak versus final validation accuracy (%). Points on the diagonal kept their peak. Color is starter.

Fourteen of fifteen trials stay near the diagonal. Trial 12 (`mid`, `slope_2deg`, seed `111`) peaked at `64.55%` and finished at `41.04%`. That peak still entered the grade tables. The second `2°` trial (t15, `narrow`) did not collapse: peak `55.38%`, final `54.18%`.

### What do the axis grades say?

Each unused value starts at grade `0.5`. After a trial, `raw` is the mean peak `val_acc` of scored combos with that value. Then `grade = 0.7 * old + 0.3 * raw`. Grades mix the axis of interest with whatever else those trials used.

![Per-axis search grades](/assets/experiments/008-axis-grades.png)

> [!CAPTION] Figure 3. Current EMA grades by group. The gray line is the unused-value start `0.5`.

`deep` (`0.691`) still leads `starter`. `mid` dropped to `0.680` after t14. `narrow` is `0.552`. `always` (`0.653`) is close to `slope_3deg` (`0.645`). `slope_2deg` rose to `0.597` after the second, non-collapsed trial. `greedy` has the highest algorithm grade (`0.688`). `composed_cosine` and `simulation_epochs=10` have never been drawn.

### Did the networks grow?

Start parameter counts are the four sequential starters. Final counts come from executed add/delete actions.

![Start and final parameter counts by trial](/assets/experiments/008-param-growth.png)

> [!CAPTION] Figure 4. Start (gray) and final (blue) parameter counts for each scored trial.

Fourteen trials grew. Trial 12 still shrank from `140389` to `11149`. Trial 14 (`mid` / `3°`) grew to `155644`. Trial 15 (`narrow` / `2°`) grew only to `34446`.

### Does the slope gate still run search on CIFAR?

Count simulation JSON files and executed timeline actions. `always` should search almost every generation except the last. A slope gate that never opens would sit near zero.

![Simulations run versus actions executed](/assets/experiments/008-search-activity-by-scheduler.png)

> [!CAPTION] Figure 5. Mean simulation calls and executed actions by scheduler. Gray markers are individual trials.

`always` runs about `9` searches on `10`-generation cells and about `19` on `20`-generation cells. `slope_3deg` ran `18` searches on both `20`-generation trials and `8` on the new `10`-generation trial (t14). `slope_2deg` ran `7` searches on t12 and `8` on t15. The 3° gate fires on CIFAR in this sample. It is not a no-search control.

### What actions ran?

![Executed action mix by trial](/assets/experiments/008-action-composition.png)

> [!CAPTION] Figure 6. Executed action counts by short label for each scored trial.

Add sequential linear, residual conv, residual linear, dropout, and delete all appear. No trial is a fixed-architecture SGD control. Neuron-resize stays off, so width changes come only from layer add/delete.

## Grouped final results

Peak `val_acc` is the search score. Final train/val are the last recorded epoch. Mean peak by starter: `deep` `69.03%` (`n=5`), `mid` `67.26%` (`n=4`), `base` `65.65%` (`n=2`), `narrow` `55.30%` (`n=4`).

| Trial | Seed | Starter | Alg | LR | `lr_alpha` | Sched | Peak val (%) | Final val (%) | Final train (%) | Start params | Final params | Sims | Acts |
| --- | ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 100 | deep | sequential_halving_beam | exponential | 0.03 | always | 70.94 | 69.85 | 66.11 | 199914 | 209322 | 9 | 9 |
| 2 | 101 | deep | montecarlo | exponential | 0.03 | always | 68.47 | 67.13 | 63.31 | 199914 | 232458 | 9 | 9 |
| 3 | 102 | base | best_first | step | 0.03 | 3° | 67.29 | 66.81 | 61.62 | 79060 | 82024 | 18 | 18 |
| 4 | 103 | mid | greedy | step | 0.03 | 3° | 69.84 | 69.36 | 65.71 | 140389 | 154879 | 18 | 18 |
| 5 | 104 | narrow | sequential_halving_beam | exponential | 0.01 | 3° | 57.13 | 55.81 | 49.56 | 33390 | 39726 | 18 | 18 |
| 6 | 105 | narrow | montecarlo | step | 0.03 | always | 56.33 | 54.85 | 46.94 | 33390 | 39854 | 19 | 19 |
| 7 | 106 | deep | greedy | step | 0.001 | always | 70.91 | 70.57 | 65.63 | 199914 | 211578 | 19 | 19 |
| 8 | 107 | narrow | ugape_deepen | exponential | 0.03 | always | 52.34 | 51.63 | 47.86 | 33390 | 37742 | 9 | 9 |
| 9 | 108 | deep | greedy | step | 0.001 | always | 66.80 | 64.16 | 61.56 | 199914 | 206970 | 9 | 9 |
| 10 | 109 | mid | montecarlo | exponential | 0.001 | always | 70.84 | 70.84 | 66.32 | 140389 | 159784 | 19 | 19 |
| 11 | 110 | base | montecarlo | step | 0.01 | always | 64.00 | 63.64 | 56.59 | 79060 | 84798 | 9 | 9 |
| 12 | 111 | mid | montecarlo | exponential | 0.001 | 2° | 64.55 | 41.04 | 42.03 | 140389 | 11149 | 7 | 7 |
| 13 | 112 | deep | greedy | exponential | 0.001 | always | 68.03 | 68.03 | 61.64 | 199914 | 202266 | 9 | 9 |
| 14 | 113 | mid | best_first | step | 0.03 | 3° | 63.79 | 62.93 | 55.28 | 140389 | 155644 | 8 | 8 |
| 15 | 114 | narrow | best_first | exponential | 0.03 | 2° | 55.38 | 54.18 | 50.13 | 33390 | 34446 | 8 | 8 |

Best combo (trial 1):

| Axis | Value |
| --- | --- |
| `starter` | `deep` (16/48, `199914` params) |
| `epochs` | `10` |
| `generations` | `10` |
| `simulation_alg` | `sequential_halving_beam` |
| `lr_schedule` | `composed_exponential` |
| `lr_alpha` | `0.03` |
| `simulation_time` | `60` s |
| `simulation_epochs` | `20` |
| `simulation_set_size` | `100` |
| `simulation_scheduler` | `always` |

Trial 7 is `0.03` percentage points behind with `greedy` + `composed_step` + `lr_alpha=0.001` + `20` generations. Do not treat trial 1 as unique.

## Training-history analysis

Curves have different lengths because `epochs × generations` is `50`, `100`, `200`, or `400`.

![Training accuracy curves by starter](/assets/experiments/008-training-curves.png)

> [!CAPTION] Figure 7. Training accuracy (%) over epochs for every scored trial. Color is starter.

![Validation accuracy curves by starter](/assets/experiments/008-validation-curves.png)

> [!CAPTION] Figure 8. Validation accuracy (%) over epochs. Color is starter. The CIFAR-10 test split is logged as `val_acc`.

`deep` and `mid` climb into the high `60s` / low `70s` within about `50` epochs, then jitter on a plateau. `narrow` flattens near `55%` even when the run lasts `400` epochs.

Trial 12 is still the break. Peak `64.55%` is at global epoch `98`, still inside generation 4, with `150199` parameters. The delete at the end of generation 4 drops the net to `11914` parameters. Generation 5 then starts at `35.08%` validation accuracy. Two later linear adds only recover to `41.04%`. The search still stored the pre-delete peak.

Trial 14 (`mid`, `5` epochs, `10` generations, `best_first`, `3°`) is a short run. It never deletes. Accuracy rises, then sits near `63%`. Trial 15 (`narrow`, `2°`) stays on the low `narrow` plateau and does not repeat the t12 collapse.

Training accuracy sits below validation accuracy on almost every trial. Train uses RandomCrop and horizontal flip. Val is the unaugmented test split.

## Limitations and seed effects

- Fifteen trials cannot cover `87480` combos. Grades on rare values stay near `0.5` if those values are never sampled.
- One seed per combo. The `0.03` percentage-point gap between t1 and t7 can be seed luck.
- Peak `val_acc` is the official CIFAR-10 test split. It is not a hidden validation set. A collapse after the peak still scores the peak.
- Axis grades confound the named value with the rest of the combo. `ugape_deepen` has one trial, and that trial is `narrow`.
- `composed_cosine` and `simulation_epochs=10` have no evidence yet.
- There is no `never`-search control in this pool.
- Neuron-resize and new simulation-set generators are out because Experiments 006 and 007 are unfinished.
- Sequential `MinimalCifarNet` is not a published CIFAR ResNet baseline.

## Conclusions

These statements use the `15` scored trials only. They are not a finished keep-set.

1. Best combo so far is still trial 1: `deep` + `sequential_halving_beam` + `composed_exponential` + `always` + `10`/`10` + `lr_alpha=0.03` + `60` s, peak val `70.94%`.
2. Starter width is the clearest split. `deep` mean peak is `69.03%`. `mid` fell to `67.26%` after t14. `narrow` stays near `55%`.
3. The 3° slope gate does fire on CIFAR. `always` is not required for search to run. A second `2°` trial (t15) did not collapse, so the t12 shrink is not a general `2°` failure.
4. The MNIST keep pieces `sequential_halving_beam` and `composed_exponential` appear in the current best cell, but that cell uses `always`, `deep`, `lr_alpha=0.03`, `60` s, and sim-set `100`, not the MNIST 3° / `0.01` / `120` s package.
5. Do not lock a CIFAR default until the remaining `35` trials finish, and until the winner is repeated on extra seeds.

## Next experiments

1. Finish the `50`-trial search. Re-run this page and the chart script from `adaptive_search.json`.
2. Confirm the winning combo on seeds `101` and `102`.
3. After Experiment 006 finishes, re-test neuron-resize on the winning CIFAR cell, not on MNIST `big`.
4. After Experiment 007 finishes, re-test simulation-set generators on that cell.
5. Add a `never`-search control on the winning starter if a later experiment needs to ask whether search helps.
