# Experiment 000: Action-aware learning rate and slope decisions

GrowingNN changes the network while it is training. Two schedulers control this process.

1. The learning-rate scheduler controls the size of each optimizer step.
2. The simulation scheduler decides when MCTS may search for an architecture action.

This experiment tests how these schedulers work together. It uses only:

`experiments/output/train_mnist/runs/lr_scheduler_slope_angle_experiment`

Script: `experiments/train_mnist_lr_schedulers.py` — created 2026-07-18 23:16, last updated 2026-07-27 17:57.

Experiment runtime: oldest output 2026-07-27 17:58, newest output 2026-07-28 09:27 — about `15 hours 28 minutes`.

## Experiment parameters

Three parameters change across the grid.

| Parameter | Tested values | Purpose |
| --- | --- | --- |
| Slope threshold | `1°`, `3°` | Controls when the training curve is flat enough to run MCTS |
| LR warmup shape | cosine, logistic, exponential | Controls how LR returns to its target after an action |
| Random seed | `1`, `2` | Measures how much the result changes with initialization and random search |

The remaining parameters stay fixed.

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Runs | `12` | `2 thresholds × 3 warmups × 2 seeds` |
| Generations | `10` | Ten training and architecture-decision cycles |
| Simulation scheduler | `SlopeEstimationSimulationScheduler` | Runs MCTS when the absolute training-slope angle enters the configured threshold |
| Configured epochs per generation | `10` | Value passed by `train_mnist_lr_schedulers.py` |
| Actual recorded epochs per generation | `11` | Historical run used `range(epochs + 1)` |
| Recorded epochs per run | `110` | `10 generations × 11 recorded epochs` |
| Target LR | `0.01` | User-set `lr_alpha` for this experiment |
| Minimum LR | `0.001` | Global LR floor |
| Warmup length | `10` | Configured scheduler iterations after an action |
| Batch size | `64` | Training samples per batch |
| Starting parameter count | `420` | Same initial graph shape in every run |
| Simulation time limit | `500 s` | MCTS time budget |
| Simulation training epochs | `15` | Training budget inside simulation |
| Simulation set size | `2000` | Samples used by simulation |

Experiment 000 recorded `11` epochs because `gradient_descent()` used `range(epochs + 1)`. The loop now uses `range(epochs)`, so future runs record the configured `10` epochs.

## Why this experiment exists

The old `ProgressiveParabolicSchedule` changed LR inside every generation. LR started near its minimum, rose close to the target, and returned near the minimum before the generation ended.

The LR reached a low value at the end of each generation. An architecture action therefore happened near a low LR. This reduced the size of the first updates around the graph change.

The same behavior also caused problems:

- training slowed near every generation end
- the next generation started slowly even when no action occurred
- training accuracy contained LR-driven waves
- a flat training slope could mean low LR instead of real stagnation
- the simulation scheduler could make a wrong decision because LR changed the accuracy history

![Previous cyclic LR and the new action-aware warmup](/assets/experiments/000-learning-rate-design.png)

> [!CAPTION] Figure 1. The red upper curve is the old schedule. It cycles once per generation. Red crosses mark action points near the minimum LR. The lower curves are the three action-aware warmups. They remain at the target LR and restart after a structure change.

The new design removes the automatic LR cycle. It uses `WarmupSchedule.iterations_since_change`. `train_generations()` calls `structure_changed()` only after an action executes.

If no action occurs, LR remains at the target value of `0.01` used in this experiment.

An action can still disturb training. The graph changes. New parameters may appear. Existing paths may be removed. The next updates must adapt to the new graph.

![Conceptual action shock and recovery](/assets/experiments/000-instability-risk.png)

> [!CAPTION] Figure 2. This conceptual drawing shows the expected risk after an architecture change. It is not measured data. The green line shows stable training. The red line shows a possible accuracy drop and unstable recovery.

## Simulation decisions

This experiment uses `SlopeEstimationSimulationScheduler`. It reads the first, middle, and last training accuracy from one generation. With three equally spaced values, the middle value cancels from the fitted slope. The effective calculation is:

```text
angle = degrees(atan((end_training_accuracy - start_training_accuracy) / 2))
```

Simulation runs when the absolute angle is no larger than the threshold.

### One complete training timeline

The `3°` logistic seed `1` run is used as the example because it belongs to the best tested configuration and contains both skipped and executed decisions.

![Training accuracy and LR for one representative run](/assets/experiments/000-representative-timeline.png)

> [!CAPTION] Figure 3. This run uses the `3°` threshold, logistic warmup, and seed `1`. Blue is training accuracy. Orange is LR. Each red dashed line is an executed architecture action.

Conclusion from Figure 3: LR stays at the user-set target between actions. It restarts only after the red action lines. The training disturbance and LR restart occur at the same point, so they are still causally mixed.

### How one slope decision is made

![Slope decision by generation](/assets/experiments/000-slope-decisions.png)

> [!CAPTION] Figure 4. Each bar is the signed training-accuracy slope angle for one generation of the same run. The green area from `-3°` to `+3°` is the simulation zone. `A` marks an executed action. Gray bars skip simulation.

Conclusion from Figure 4: the scheduler follows the threshold. It skips generations `0`, `1`, `3`, and `8`. It runs MCTS in the other generations. This graph shows the decision directly in degrees. It does not normalize the result into an abstract ratio.

Across the full grid:

| Threshold | Simulations | Executed actions | Meaning |
| ---: | ---: | ---: | --- |
| `1°` | `30` | `30` | Requires a flatter training curve |
| `3°` | `38` | `38` | Allows earlier and more frequent search |

Every simulation returned an action in this experiment.

## When actions occur

![Executed actions by generation](/assets/experiments/000-actions-by-generation.png)

> [!CAPTION] Figure 5. Each bar counts actions at one generation across all `12` runs. Blue actions use the `1°` threshold. Green actions use the `3°` threshold.

The `3°` threshold starts earlier and produces `8` more actions than `1°`. From generation `5`, actions occur in most runs. This means the measured slope often returns to the simulation zone. It does not prove that the model has reached its best possible accuracy.

Every run also executes a generation-`9` action at global epoch `110`. The history ends at epoch `109`, and there is no generation `10`. Therefore, `12` final actions have no later metric. Final accuracy describes the graph before this last mutation.

## Accuracy gain after architecture actions

The next question is whether actions create a lasting accuracy gain. For each observable action, compare accuracy immediately before the action with accuracy at the end of the next generation. This gives the changed graph one generation to recover.

### Action order

![Validation change by action order](/assets/experiments/000-action-order.png)

> [!CAPTION] Figure 6. Bars show the mean validation-accuracy change after the first, second, third, fourth, and fifth-or-later action. Dots show individual actions. Values are percentage-point changes.

| Action order | Observed actions | Mean next-generation validation change |
| --- | ---: | ---: |
| First | `12` | `+14.96 percentage points` |
| Second | `12` | `+1.87 percentage points` |
| Third | `12` | `+2.22 percentage points` |
| Fourth | `11` | `+1.12 percentage points` |
| Fifth or later | `9` | `-0.13 percentage points` |

One early action produces most of the measured gain, while later actions add little and are harmful in `18/44` measured cases. A controlled stop-after-first-action run is still needed because first actions occur at lower accuracy and use different action types.

### Action type

The same recovery-window comparison can show which action types improve training and validation accuracy.

![Validation change by action type](/assets/experiments/000-action-types.png)

> [!CAPTION] Figure 7. Blue shows training-accuracy change and green shows validation-accuracy change over the next generation. Bars are means. Colored dots are individual actions. Values are percentage-point changes.

| Action type | Observed actions | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Add residual convolution | `12` | `+19.11 percentage points` | `+18.66 percentage points` |
| Add sequential linear | `21` | `+1.43 percentage points` | `+2.31 percentage points` |
| Add residual linear | `3` | `+5.52 percentage points` | `+1.74 percentage points` |
| Add sequential dropout | `12` | `-8.11 percentage points` | `-1.38 percentage points` |
| Delete layer | `8` | `-3.80 percentage points` | `-2.69 percentage points` |

Residual-convolution additions improve both metrics most. Dropout and deletion reduce both metrics on average. These are observational results because MCTS chooses actions at different model states.

## Final results

The final comparison uses training accuracy, validation accuracy, peak validation accuracy, and action count.

### Result grouped by slope threshold

| Threshold | Mean final training accuracy | Mean final validation accuracy | Mean peak validation accuracy | Mean actions |
| ---: | ---: | ---: | ---: | ---: |
| `1°` | `78.19%` | `81.21%` | `83.70%` | `5.00` |
| `3°` | `73.93%` | `83.91%` | `87.60%` | `6.33` |

Based on validation accuracy, `3°` is the better threshold candidate. It reaches a higher final and peak result, but uses more actions and has lower final training accuracy.

### Result grouped by warmup

| Warmup | Mean final training accuracy | Mean final validation accuracy | Mean peak validation accuracy | Mean actions |
| --- | ---: | ---: | ---: | ---: |
| Cosine | `72.73%` | `78.69%` | `86.05%` | `5.25` |
| Logistic | `84.34%` | `88.28%` | `89.55%` | `5.25` |
| Exponential | `71.10%` | `80.70%` | `81.36%` | `6.50` |

Logistic has the best mean final training and validation accuracy. It does not use more actions than cosine.

### Full configuration result

![Mean final training and validation accuracy](/assets/experiments/000-final-accuracy.png)

> [!CAPTION] Figure 8. Each pair of bars is one threshold and warmup configuration. Blue is mean final training accuracy. Green is mean final validation accuracy. Each bar averages seeds `1` and `2`.

| Threshold | Warmup | Mean final training | Mean final validation | Mean peak validation | Mean actions |
| ---: | --- | ---: | ---: | ---: | ---: |
| `1°` | Cosine | `78.31%` | `77.68%` | `83.08%` | `4.5` |
| `1°` | Logistic | `82.37%` | `82.91%` | `84.48%` | `5.0` |
| `1°` | Exponential | `73.88%` | `83.03%` | `83.56%` | `5.5` |
| `3°` | Cosine | `67.16%` | `79.70%` | `89.03%` | `6.0` |
| `3°` | Logistic | `86.32%` | `93.64%` | `94.61%` | `5.5` |
| `3°` | Exponential | `68.32%` | `78.38%` | `79.16%` | `7.5` |

## Training histories

![Training-accuracy curves for all configurations](/assets/experiments/000-training-curves.png)

> [!CAPTION] Figure 9. Each panel shows training accuracy for one threshold and warmup pair. Blue is seed `1`. Green is seed `2`. The slope scheduler makes its decisions from these curves.

The curve shapes show:

1. matched seed curves rise similarly before architecture search, then separate after different actions
2. the first large jumps occur later in the `1°` panels than in the `3°` panels
3. action points create sharp jumps or drops, while action-free sections are smoother
4. the `3°` cosine seed `2` curve has a large late drop and does not recover its earlier level
5. the `3°` exponential curves contain repeated drops and short recoveries, which matches their high action count

## Starting model and seed effect

Three seed and initialization facts limit the result:

- Every run starts with the same `420`-parameter graph shape, but the seeds create different initial weights and search paths.
- Generation-`0` validation accuracy ranges from about `24.7%` to `35.5%`. Runs with the same seed and warmup remain close until their first different architecture decision.
- The `1°` cells have final validation gaps above `21` percentage points between two seeds. Two seeds are not enough to establish reliability.

The current data does not determine whether the starting graph should be smaller. A controlled test should compare smaller graphs with the current `420`-parameter graph. Each matched run should use the same seed and initial weights.

## Immediate training disturbance after an action

This final analysis checks the short shock that appears directly after an action. It is different from the one-generation gain measured above.

For each generation change, compare the last training accuracy before the change with the first training accuracy after it. The `12` runs provide `108` comparisons.

The absolute change measures disturbance size.

| Previous generation ended with | Comparisons | Mean absolute training-accuracy change |
| --- | ---: | ---: |
| No action | `52` | `1.09 percentage points` |
| Action with cosine warmup | `17` | `7.64 percentage points` |
| Action with logistic warmup | `17` | `6.28 percentage points` |
| Action with exponential warmup | `22` | `5.79 percentage points` |

![Immediate disturbance between generations](/assets/experiments/000-generation-transition.png)

> [!CAPTION] Figure 10. The gray bar shows generation changes without an action. The other bars show changes after an action, grouped by LR warmup.

All three action groups are much less stable than the no-action group. Cosine has the largest absolute disturbance.

### Direction of the immediate change

Absolute values hide whether accuracy increased or decreased. The signed mean and drop count show the direction.

![Signed training-accuracy change by LR warmup](/assets/experiments/000-signed-generation-transition.png)

> [!CAPTION] Figure 11. Positive bars mean training accuracy increased at the generation change. Negative bars mean it decreased. Each label also gives the number of negative changes.

| Previous generation ended with | Mean signed change | Negative changes |
| --- | ---: | ---: |
| No action | `+1.09 percentage points` | `3/52` |
| Action with cosine warmup | `-4.92 percentage points` | `7/17` |
| Action with logistic warmup | `-2.64 percentage points` | `5/17` |
| Action with exponential warmup | `-4.31 percentage points` | `13/22` |

Logistic has the smallest mean drop after actions. Exponential has the smallest absolute movement but the highest drop frequency. Mutation type and LR reset still occur together, so this comparison is not causal.

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. It can be removed without breaking the rendered website, but the original board files will no longer be available.

The report now keeps three trackable artifacts:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- a `404 KB` normalized data snapshot at `documentation/website/data/experiments/experiment-000-slope-angle-lr-warmup.json`

`generate_experiment_charts.py` updates the snapshot when raw output exists. If raw output is missing, it reads the snapshot instead. The page and charts will therefore remain reproducible after the raw experiment folder is deleted.

These documentation artifacts are not ignored, but they are currently untracked. They must be included in a Git commit before the raw output is removed from this machine.

## Conclusions

1. Action-aware warmup removes the old LR wave at no-action generation transitions.
2. The simulation scheduler now reads a cleaner training history.
3. Architecture actions remain the largest source of immediate disturbance.
4. The first action provides most of the measured gain.
5. Residual-convolution additions provide the largest observed action-type gain.
6. Later dropout and deletion actions are often harmful.
7. `3°` logistic is the best tested pair, but only two seeds support it.
8. Every run changes the graph after its last recorded metric, so the saved final graph has no measured post-change accuracy.
9. The experiment cannot separate mutation effects from LR-reset effects.

## Next steps

### 1. Fix the experiment protocol

- `gradient_descent()` now runs exactly the configured number of epochs. Experiment 000 itself still contains `11` historical records per generation.
- Do not search after the final training generation. Experiment 000 executed generation-`9` actions after saving its last metrics. `train_generations()` now skips simulation when no recovery generation remains.
- The runner already seeded PyTorch and CUDA before model creation. It now also seeds Python and NumPy through `seed_experiment()`. Matched policies with the same seed should start from the same weights and random streams.
- Run one identical configuration twice. If results differ, record a starting-weight hash and enable deterministic CUDA operations where supported.

### 2. Stabilize the next experiment

Three protocol changes are now in place:

- `seed_experiment()` resets Python, NumPy, PyTorch, and CUDA for every run.
- `train_generations()` does not search after the final training generation.
- `gradient_descent()` records exactly the configured number of epochs.

The next script also requests deterministic PyTorch algorithms and disables cuDNN benchmarking.

### 3. Recommended next experiment: logistic recovery stability

The next question is how long logistic LR recovery should last after an architecture change. The experiment uses a smaller fixed starting graph and changes only warmup length.

The script is:

`experiments/train_mnist_lr_stability.py`

| Parameter | Value | Reason |
| --- | ---: | --- |
| Starting graph | `158` parameters | Uses `channels=2` and `hidden=8` as a smaller baseline |
| LR warmup | logistic | Best final result in Experiment 000 |
| Warmup length | `5`, `10`, `20` epochs | Tests fast, current, and slow recovery |
| Slope threshold | `3°` | Best tested threshold candidate |
| Seeds | `1`, `2`, `3`, `4` | Four matched seeds across all three warmups |
| Runs | `12` | `3 warmups × 4 seeds` |
| Generations | `10` | Same architecture-decision budget |
| Epochs per generation | `10` | Now produces exactly `10` recorded epochs |
| Target LR | `0.01` | Preserves the previous target |
| MCTS time | `500 s` | Preserves search quality from Experiment 000 |
| Simulation epochs | `15` | Preserves the simulation training budget |
| Simulation set size | `2000` | Preserves the simulation sample budget |

Measure:

- seed spread in final training and validation accuracy
- absolute and signed accuracy change after actions
- epochs needed to recover the pre-action accuracy
- peak-to-final loss
- action count and first-action generation

The preferred warmup should have low seed spread, a small signed post-action drop, and no loss in final validation accuracy.

Run it with:

```text
python experiments/train_mnist_lr_stability.py
```

This script has not been run yet.

### 4. After the stability experiment

Repeat the best configuration with the same seed in a separate output folder. This checks exact reproducibility. Only then compare another slope threshold or implement the action-only versus LR-reset-only causal test.

The chart script reads the current board files:

```text
python documentation/website/scripts/generate_experiment_charts.py
```
