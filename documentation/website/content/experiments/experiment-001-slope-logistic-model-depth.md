# Experiment 001: Slope scheduling across starting model depth

GrowingNN changes the network while it is training. Experiment 000 found that `3°` slope scheduling with logistic LR warmup was the strongest tested pair on a `420`-parameter MNIST starter graph.

This experiment asks whether that result still holds when the starting graph is thinner. Neuron add and delete actions stay disabled, so capacity is reduced by removing initial layers instead of narrowing channels.

Raw output:

`experiments/output/train_mnist/runs/exp001_slope_logistic_model_depth`

Script: `experiments/train_mnist_exp001_slope_model_depth.py` — created 2026-07-28 23:50.

Experiment runtime: oldest board output 2026-07-29 01:15, newest output 2026-07-30 01:15 — about `24 hours`. Recorded training time across all runs is about `25 hours 24 minutes`.

## Experiment parameters

Two parameters change across the grid. LR warmup stays fixed to logistic.

| Parameter | Tested values | Purpose |
| --- | --- | --- |
| Slope threshold | `2°`, `3°`, `4°` | Controls when the training curve is flat enough to run MCTS |
| Starting model depth | big, medium, very small | Changes initial capacity by removing layers |
| Random seed | `100`, `101` | Measures seed spread under the deterministic seeding protocol |

The remaining parameters stay fixed.

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Runs | `18` | `3 thresholds × 3 depths × 2 seeds` |
| Completed runs | `18` | Full grid |
| Generations | `10` | Ten training and architecture-decision cycles |
| Simulation scheduler | `SlopeEstimationSimulationScheduler` | Runs MCTS when the absolute training-slope angle enters the configured threshold |
| LR warmup | logistic | Best warmup from Experiment 000 |
| Warmup length | `10` | Scheduler iterations after an action |
| Warmup steepness `k` | `10` | Logistic shape parameter |
| Configured epochs per generation | `10` | Produces exactly `10` recorded epochs |
| Recorded epochs per run | `100` | `10 generations × 10 recorded epochs` |
| Target LR | `0.01` | User-set `lr_alpha` |
| Minimum LR | `0.001` | Global LR floor |
| Batch size | `64` | Training samples per batch |
| Simulation time limit | `500 s` | MCTS time budget |
| Simulation training epochs | `15` | Training budget inside simulation |
| Simulation set size | `2000` | Samples used by simulation |

### Starting graphs

| Depth label | Initial layers | Starting parameters | Construction |
| --- | --- | ---: | --- |
| Big | `conv1`, `conv2`, `linear`, `linear2` | `420` | Same stem as Experiment 000 |
| Medium | `conv1`, `linear`, `linear2` | `276` | Experiment 000 stem with `conv2` removed |
| Very small | `conv1`, `linear2` | `76` | Experiment 000 stem with `conv2` and the hidden linear removed |

### Protocol changes already applied before this experiment

- `gradient_descent()` records exactly `10` epochs per generation
- `can_simulate()` refuses the last generation, so there is no unevaluated terminal mutation
- matched seeds are `100` and `101`
- `configure_deterministic_seeding()` requests deterministic PyTorch algorithms and disables cuDNN benchmarking

cuDNN benchmarking is a CUDA convolution shortcut. When it is on, cuDNN may try several convolution algorithms and keep the fastest one for the current shapes. That can make training faster after a short search cost, especially for larger convolutions. It can also change the chosen kernel between runs, so the same seed may not follow the same compute path. This experiment turns benchmarking off to keep matched seeds comparable. The tradeoff is less automatic speed tuning.

## Why this experiment exists

Experiment 000 kept one starting graph shape at `420` parameters, and the first action produced almost all of the measured gain. A thinner starter should need more growth steps, so this page measures next-generation action gains and splits them by starting depth.

## One complete training timeline

The example run uses the `2°` threshold, the big starter, and seed `100`. It is selected because it has the highest final validation accuracy in the full grid.

![Training accuracy and LR for one representative run](/assets/experiments/001-representative-timeline.png)

> [!CAPTION] Figure 1. Blue is training accuracy. Orange is LR. Each red dashed line is an executed architecture action. The run uses `2°`, logistic warmup, the big starter, and seed `100`.

Conclusion from Figure 1: LR stays at the target between actions and restarts only after structure changes. The first residual-convolution action creates the largest rise.

## Slope decisions across the full grid

The slope scheduler does not read raw accuracy. It reads how fast training accuracy is changing inside one generation.

For one generation it takes the first and last training accuracy, forms the fitted slope, and converts that slope to degrees:

```text
slope_angle_degrees = degrees(atan((train_acc_end - train_acc_start) / 2))
```

The sign shows direction. A positive angle means training accuracy rose. A negative angle means it fell. The size of the angle shows how steep that rise or fall was. Simulation runs when the absolute angle is no larger than the chosen threshold.

The next chart is not one example run. Each panel is one starting depth. Thin colored lines are the mean slope angle for each threshold (`2°`, `3°`, `4°`). The thick black line is the mean over all runs of that depth.

![Mean slope angles by generation](/assets/experiments/001-slope-decisions.png)

> [!CAPTION] Figure 2. Y-axis values are slope angles in degrees, not accuracy percentages. Thin blue, green, and orange lines are means for `2°`, `3°`, and `4°`. The thick black line is the mean over all six runs in that panel.

Conclusion from Figure 2:

1. Big and medium start with steeper training slopes. Early learning is still moving quickly.
2. Very small starts flatter. Its early training curve rises more slowly, so search can start sooner.
3. From about generation `3`, all three depths move toward flatter angles. That matches the later rise in action counts.
4. The threshold lines are not equally close in every panel. On very small they stay close. On big they spread more.
5. A flatter late curve does not only mean “ready to search.” It can also mean the run already found a strong structure, or that learning has slowed and later actions are no longer helping much.

## When actions occur

![Executed actions by generation and starting depth](/assets/experiments/001-actions-by-generation.png)

> [!CAPTION] Figure 3. Each generation has three bars, one per starting depth. Each depth contributes six completed runs: two seeds for each of the three slope thresholds. The dashed line separates generations `0–2` from later generations.

Generations `0–2` show a clear depth effect. Very small acts immediately. Medium acts a little later. Big often waits longer. That early pattern matches the assumption that a thinner starter needs earlier architecture change.

From generation `3` onward, almost every depth acts often. Those later bars look more mixed. High later action counts do not prove that later actions are useful. The recovery-window charts below test that.

| Depth | Mean amount of actions in generations `0–3` | Mean amount of actions in the full run | First-action generations |
| --- | ---: | ---: | --- |
| Big | `2.00` | `6.67` | `0`, `0`, `1`, `1`, `3`, `3` |
| Medium | `2.33` | `7.00` | `0`, `0`, `1`, `1`, `1`, `3` |
| Very small | `2.50` | `6.83` | `0`, `0`, `0`, `0`, `0`, `0` |

Very small concentrates more of its search early. Big and medium still execute many later actions after the early growth window.

## Accuracy gain after architecture actions

The next question is whether actions create a lasting accuracy gain. For each observable action, compare accuracy immediately before the action with accuracy at the end of the next generation. This gives the changed graph one generation to recover.

### Action order across the full grid

![Validation change by action order](/assets/experiments/001-action-order.png)

> [!CAPTION] Figure 5. Bars show the mean validation-accuracy change after the first, second, third, fourth, and fifth-or-later action. Dots show individual actions from all `18` runs. Values are percentage-point changes.

| Action order | Observed actions | Mean next-generation validation change |
| --- | ---: | ---: |
| First | `18` | `+29.31 percentage points` |
| Second | `18` | `+3.16 percentage points` |
| Third | `18` | `-3.28 percentage points` |
| Fourth | `18` | `+8.34 percentage points` |
| Fifth or later | `50` | `+0.37 percentage points` |

The first action dominates. The second action still helps on average. The third action is harmful on average. From the fifth action onward, the mean effect is near zero.

That shape is a structural warning. The current starters and action set do not create a smooth multi-step growth ladder. Later search often adds noise. Neuron add and delete stay disabled here because they are not stable enough yet. The next starter designs should aim for a flatter early-gain curve, with useful gains spread across about three or four early actions instead of one jump.

### Action order by starting depth

![Validation change by action order and starting depth](/assets/experiments/001-action-order-by-depth.png)

> [!CAPTION] Figure 6. Each panel uses the same next-generation validation change as Figure 5. One panel is one starting depth. Bars are means. Dots are individual actions.

| Depth | First-action mean | Second-action mean | Later-action mean | Later harmful |
| --- | ---: | ---: | ---: | ---: |
| Big | `+40.68 percentage points` (`n=6`) | `-0.16 percentage points` (`n=6`) | `+0.94 percentage points` (`n=34`) | `14/34` |
| Medium | `+35.24 percentage points` (`n=6`) | `-0.61 percentage points` (`n=6`) | `+2.16 percentage points` (`n=36`) | `15/36` |
| Very small | `+12.00 percentage points` (`n=6`) | `+10.26 percentage points` (`n=6`) | `+1.73 percentage points` (`n=35`) | `16/35` |

Big and medium put almost all useful gain into the first action. Very small is different: the first two actions both add about `10` to `12` percentage points. That early flat pattern is what the experiment wanted to see.

Very small still fails the accuracy goal. Its final validation accuracy stays near `49%`.

### Action order by slope threshold

![Validation change by action order and slope threshold](/assets/experiments/001-action-order-by-slope.png)

> [!CAPTION] Figure 7. Same recovery window as Figure 5. One panel is one slope threshold across all depths.

| Threshold | First-action mean | Second-action mean | Fifth-or-later mean |
| ---: | ---: | ---: | ---: |
| `2°` | `+27.35 percentage points` | `+0.44 percentage points` | `+0.41 percentage points` |
| `3°` | `+29.09 percentage points` | `+3.63 percentage points` | `-0.19 percentage points` |
| `4°` | `+31.48 percentage points` | `+5.42 percentage points` | `+0.92 percentage points` |

A higher threshold spreads early gain more. The second action grows from nearly zero at `2°` to about `+5.4` percentage points at `4°`. That is useful for multi-step control.

Final accuracy still favors the lower thresholds. Across all depths, `2°` has the best mean final validation accuracy, then `3°`, then `4°`. So `3°` is the practical balance: more distributed early gains than `2°`, without the weaker finals and larger post-action shock of `4°`.

### Action type

![Validation change by action type](/assets/experiments/001-action-types.png)

> [!CAPTION] Figure 8. Blue shows training-accuracy change and green shows validation-accuracy change over the next generation. Bars are means. Colored dots are individual actions.

| Action type | Observed actions | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Add residual convolution | `41` | `+15.31 percentage points` | `+15.28 percentage points` |
| Add sequential linear | `39` | `+2.85 percentage points` | `+2.20 percentage points` |
| Add residual linear | `4` | `+7.38 percentage points` | `+0.43 percentage points` |
| Add sequential dropout | `22` | `-4.43 percentage points` | `+2.42 percentage points` |
| Delete layer | `17` | `-2.37 percentage points` | `-4.08 percentage points` |

Residual convolution is the strongest observed action. Deletion is harmful on average.

One enabled action never appears: sequential convolution. Across all simulation files in this experiment, `Add Seq Conv Layer Action` was never even a candidate. It was not rejected by MCTS. It was never generated as a legal move.

Why: `AddSeqConvLayer` only opens between two sequential layers when both sides have the same 4-D feature-map shape. These MNIST starters move from convolution into pooling and then into linear layers. Most sequential bridges are therefore convolution-to-linear or linear-to-linear. Those shapes do not satisfy the sequential-convolution rule. Residual convolution uses a different insertion path, so it can still appear.

That missing action matters. The search can add residual convolution, but it cannot insert a normal sequential convolution to rebuild a thin stem step by step. That is a bug to fix: sequential convolution must become legal on these rebuild edges.

### Action type by starting depth

![Accuracy change by action type and starting depth](/assets/experiments/001-action-types-by-depth.png)

> [!CAPTION] Figure 9. One panel is one starting depth. Blue is training-accuracy change. Green is validation-accuracy change. Empty types mean that depth never executed that action.

| Depth | Dominant useful action | Notable weak or harmful pattern |
| --- | --- | --- |
| Big | residual convolution (`n=19`, mean validation `+14.50 percentage points`) | sequential linear and dropout add little |
| Medium | residual convolution (`n=14`, mean validation `+18.03 percentage points`) | sequential linear is slightly negative on average |
| Very small | sequential linear is the most common action (`n=18`); residual convolution still helps when found (`n=8`, `+12.32 percentage points`) | deletion is strongly negative (`n=6`, `-9.38 percentage points`) |

Very small often rebuilds with sequential linear layers first. It finds residual convolution less often. Medium can add residual convolution and still fails to catch the big starter.

## Final results

### Result grouped by starting depth

![Mean final training and validation accuracy by starting depth](/assets/experiments/001-final-accuracy.png)

> [!CAPTION] Figure 10. Each pair of bars is one starting depth averaged across all three slope thresholds and both seeds.

| Depth | Mean final training | Mean final validation | Mean amount of actions | Mean final parameters |
| --- | ---: | ---: | ---: | ---: |
| Big | `89.85%` | `91.20%` | `6.67` | `2671` |
| Medium | `73.62%` | `82.32%` | `7.00` | `1991` |
| Very small | `47.05%` | `48.94%` | `6.83` | `249` |

The Experiment 000 big starter remains strongest. Medium is weaker. Very small never reaches useful MNIST accuracy.

### Why thinner starters stay behind

Medium ran about as many actions as big. Very small had two useful early actions. Neither finished near the big starter.

Short answer from the boards:

- Big used residual convolution most (`19`). Very small used it least (`8`) and preferred sequential linear layers (`18`).
- Very small stayed tiny (`~249` parameters). Medium grew large (`~1991`) and still lagged, so size alone is not the whole story.
- Sequential convolution never appeared as a legal move, so thin starters could not rebuild a missing conv stem in the natural sequential way.

### Result grouped by slope threshold

![Mean final training and validation accuracy by slope threshold](/assets/experiments/001-final-accuracy-by-slope.png)

> [!CAPTION] Figure 11. Each pair of bars is one slope threshold averaged across all three depths and both seeds.

| Threshold | Mean final training | Mean final validation | Mean peak validation | Mean amount of actions |
| ---: | ---: | ---: | ---: | ---: |
| `2°` | `75.89%` | `75.39%` | `78.94%` | `5.83` |
| `3°` | `72.12%` | `75.13%` | `82.05%` | `6.83` |
| `4°` | `62.50%` | `71.94%` | `78.88%` | `7.83` |

`2°` has the best mean final validation accuracy. `4°` spreads early gains more, but finishes worse. `3°` is the balanced choice for later work.

## Final graph comparison

The board stores final graphs as wide simplified PDFs. The images below are PNG renders of those PDFs, stacked one under the other. Original PDFs are in `documentation/website/app/public/assets/experiments/exp001-graphs/`.

A shared visual issue appears on medium and very-small graphs: after residual convolution into a linear target, the residual block often contains `AdaptiveMaxPool2d` plus flatten, while the main stem still has `max_pool2d` and later `adaptive_avg_pool2d`. Future pooling experiments should compare only max pool, only average pool, and no pool.

### Starting graphs before growth

These are generation-`0` graphs for seed `100`.

![Big starter at generation 0](/assets/experiments/exp001-graphs/start-big-gen0-1.png)

> [!CAPTION] Figure 12. Big starter before growth: `conv1`, `conv2`, `linear`, `linear2`.

![Medium starter at generation 0](/assets/experiments/exp001-graphs/start-medium-gen0-1.png)

> [!CAPTION] Figure 13. Medium starter before growth: `conv1`, `linear`, `linear2`.

![Very small starter at generation 0](/assets/experiments/exp001-graphs/start-very_small-gen0-1.png)

> [!CAPTION] Figure 14. Very small starter before growth: `conv1`, `linear2`.

### Best final graph for each depth

![Best big final graph](/assets/experiments/exp001-graphs/best-big-2deg-seed100-final-1.png)

> [!CAPTION] Figure 15. Best big final graph: `2°`, seed `100`, final validation `96.41%`.

![Best medium final graph](/assets/experiments/exp001-graphs/best-medium-4deg-seed100-final-1.png)

> [!CAPTION] Figure 16. Best medium final graph: `4°`, seed `100`, final validation `89.59%`.

Figure 16 shows the core bug. The code cannot add a normal sequential convolution on these graphs. So the search tries to rebuild that missing step the hard way.

What should be one action:

- add one sequential convolution after `conv1`

What the medium run actually did:

1. Generation `0`: add residual convolution from `conv1` to `linear`
2. Generation `2`: delete `linear`
3. Generation `3`: add sequential linear from `conv1` to `linear2`
4. Generation `4`: add residual convolution from `conv1` to that sequential linear
5. Generation `5`: delete the sequential linear
6. Generations `7–8`: add sequential linear again, then residual convolution into it again

So the run went in a circle for many generations. It added and deleted layers to recreate something close to a sequential convolution. That should have been one action, not a long add/delete loop.

Figure 17 shows the same problem on very small: residual convolution appears, but the graph still does not climb cleanly toward the medium or big stem.

![Best very small final graph](/assets/experiments/exp001-graphs/best-very_small-3deg-seed101-final-1.png)

> [!CAPTION] Figure 17. Best very small final graph: `3°`, seed `101`, final validation `53.89%`.

### Final graphs under the balanced `3°` threshold

![3° big final graph](/assets/experiments/exp001-graphs/slope3-big-seed100-final-1.png)

> [!CAPTION] Figure 18. `3°` big seed `100`, final validation `94.42%`.

![3° medium final graph](/assets/experiments/exp001-graphs/slope3-medium-seed100-final-1.png)

> [!CAPTION] Figure 19. `3°` medium seed `100`, final validation `86.19%`.

Figure 19 shows the same residual workaround again: residual convolution from `conv1` into `seq_linear_0`.

It also ends with two sequential dropouts on the same edge between `seq_linear_0` and `seq_linear_1`:

- Generation `7`: dropout `p=0.5`. Simulation said accuracy after about `0.596`. MCTS visited this candidate only once.
- Generation `8`: dropout `p=0.2`. Simulation said accuracy after about `0.694`. Also only one visit.

In the same generations, residual-convolution candidates looked better in simulation (`0.82` and `0.866`). But MCTS barely tested them, so the weak dropout options won.

What “visits” means here: during search, MCTS tries a candidate a number of times. One visit is almost no check. A candidate with higher simulated accuracy should normally be checked more often before a worse candidate is chosen. Here the opposite happened.

The second dropout did raise real validation from `80.03%` to `84.00%`, so that step was not useless. Still, two dropouts on the same edge should not be a normal growth path. Search should block stacked dropouts there.

![3° very small final graph](/assets/experiments/exp001-graphs/slope3-very_small-seed101-final-1.png)

> [!CAPTION] Figure 20. `3°` very small seed `101`, final validation `53.89%`.

Figure 20 stays mostly sequential linear. Residual convolution does appear later, but early search prefers linear layers. Then the run mixes residual adds with deletes. Without a legal sequential-convolution rebuild, very small never becomes medium or big.

## Training histories

### By starting depth

![Training-accuracy curves by starting model depth](/assets/experiments/001-training-curves.png)

> [!CAPTION] Figure 21. Each panel is one starting depth. Line color marks the slope threshold.

The curve shapes show:

1. big runs rise into the high-accuracy region after the first strong residual-convolution addition
2. medium runs also rise early, then some weaken later
3. very small stays below about `60%` training accuracy for the whole run

### By slope threshold

![Training-accuracy curves by slope threshold](/assets/experiments/001-training-curves-by-slope.png)

> [!CAPTION] Figure 22. Each panel is one slope threshold. Line color marks the starting depth.

Depth still separates the curves more than threshold does. Inside each threshold, big stays high, medium is mixed, and very small stays low.

## Structures used in this experiment

Module lists below come from the chosen candidate structure in the last simulation of each run. Parameter counts come from generation boards.

| Depth | Mean modules | Mean stem conv | Mean residual conv | Mean sequential linear | Mean dropout | Mean final parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Big | `8.83` | `1.83` | `3.17` | `1.50` | `0.83` | `2671` |
| Medium | `7.83` | `1.00` | `2.17` | `1.83` | `1.67` | `1991` |
| Very small | `6.50` | `1.00` | `0.83` | `2.17` | `1.17` | `249` |

Column meanings:

- Mean modules: average number of named modules in the final chosen structure
- Mean stem conv: average count of original stem convolutions such as `conv1` or `conv2`
- Mean residual conv: average count of `res_conv_*` modules
- Mean sequential linear: average count of `seq_linear_*` modules
- Mean dropout: average count of `seq_dropout_*` modules
- Mean final parameters: average trainable parameter count at the end of the run

Example best finals:

| Depth | Best run | Final validation | Final modules |
| --- | --- | ---: | --- |
| Big | `2°` seed `100` | `96.41%` | `conv1`, `conv2`, five residual convolutions, one sequential linear, `linear2` |
| Medium | `4°` seed `100` | `89.59%` | `conv1`, two residual convolutions, one sequential linear, `linear2` |
| Very small | `3°` seed `101` | `53.89%` | `conv1`, three sequential linears, one dropout, one residual convolution, `linear2` |

### Proposed starters for the next experiment

Next we want early growth with a few useful steps, not one huge jump and then noise. The table is a short list of starters to try. We do not need to run all of them at once.

| Priority | Proposed starter | What changes | Why try it |
| ---: | --- | --- | --- |
| 1 | Medium, smaller first residual (`ch=4`, `h=4`) | Same medium depth, weaker first residual gain | Less one-step jump |
| 2 | Medium, thinner stem (`ch=2`, `h=8`) | Thinner first layers | Check if width, not only depth, causes the jump |
| 3 | Big stem, thinner head (`ch=2`, `h=8`) | Keep two conv places, smaller head | Keep the strong big attachment points |
| 4 | Gap model: `conv1 → linear → linear2` | Between very small and medium size | Fill the missing middle starter |
| 5 | Current medium, no stacked dropout / less late delete | Same medium graph, safer late actions | Test search rules after Figure 19 |
| 6 | Current very small, prefer residual conv early | Same very small graph, different action preference | Check if early residual search helps |
| 7 | Fix sequential convolution on rebuild edges | Same thin starters, legal sequential conv | Let small become medium, and medium become big |
| 8 | Only max pool | Same starter family, pooling changed | Compare pooling setups |
| 9 | Only average pool | Same starter family, pooling changed | Compare pooling setups |
| 10 | No pool | Same starter family, pooling removed | Compare pooling setups |

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. The report keeps:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- copied final and start graph PDFs and PNG renders under `documentation/website/app/public/assets/experiments/exp001-graphs/`
- a normalized data snapshot at `documentation/website/data/experiments/experiment-001-slope-logistic-model-depth.json`

`generate_experiment_001_charts.py` updates the snapshot when raw output exists. If raw output is missing, it reads the snapshot instead.

These documentation artifacts must be committed before the raw experiment folder is removed from this machine.

## Conclusions

1. Search timing depends on starter size. Very small starts architecture search early. Big starts later.
2. The big starter ends best. Medium is weaker. Very small stays too weak for MNIST in this setup.
3. Slope threshold has two messages. Action-order by slope shows that a higher threshold spreads gain better across early actions. Final means show that `2°` has the best average validation accuracy. That is why `3°` is the balanced choice: better stability than `2°`, better finals than `4°`.
4. Big and medium get almost all useful gain from the first action. Very small gets useful gain from the first two actions, then stops improving much.
5. Residual convolution is the strongest action type. Sequential convolution never appears as a legal move on these MNIST graphs.
6. Because sequential convolution is missing, medium spends many generations adding and deleting layers to fake that rebuild. That long loop should have been one sequential-convolution action.
7. Medium can grow in size and still fail to become the big architecture. Very small can take early useful steps and still fail to become the medium architecture. That is an error to fix.
8. Under `3°`, medium added two dropouts on the same edge. MCTS barely checked better residual-convolution options. “Visits” means how many times search tested a candidate. One visit is almost no check. A better candidate should be tested more before a weaker one wins.
9. Those two dropouts did raise real validation a bit, but stacking dropouts on one edge should not be allowed.
10. Medium and very-small graphs mix normal stem pooling with residual adaptive max pooling. Future runs should compare only max pool, only average pool, and no pool.

## Next steps

1. Fix the bug that blocks sequential convolution in the cases shown in this report, so a small model can become medium and a medium model can become big.
2. Keep `3°` logistic as the default pair while that fix is tested.
3. Fix stacked dropout: the same edge should not get multiple sequential dropout actions.
4. After those fixes, test the starter and pooling ideas in the table above.
