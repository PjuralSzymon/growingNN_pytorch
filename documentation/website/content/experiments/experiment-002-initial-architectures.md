# Experiment 002: Initial architectures

GrowingNN changes the network while it is training. Experiment 000 chose action-aware LR warmup. Experiment 001 chose `3°` slope scheduling with logistic warmup as the practical balance.

Experiment 001 also showed that the accuracy gain from architecture actions was not spread evenly across generations. Most useful gain came from the first action. Later actions added noise. The main structural cause was incomplete sequential-convolution insertion: thin starters could not rebuild a missing stem convolution in the natural sequential way, so search used residual workarounds, deletes, and stacked dropouts instead.

That insertion path is now fixed through `AddSeqConvLayer.try_build_eye_convolution_for_insert_before_flatten`. Experiment 002 keeps the Experiment 001 scheduler pair fixed and varies only the starting architecture. The goal is to see which initial graphs grow usefully for later experiments.

Raw output:

`experiments/output/train_mnist/runs/exp002_initial_architectures`

Script: `experiments/train_mnist_exp002_initial_architectures.py` — created 2026-08-04 14:38.

Experiment runtime so far: oldest board start `2026-08-04T12:40:26Z`, newest update among available boards `2026-08-04T19:48:29Z`. Recorded training time across the loaded runs is about `14 hours`. The grid is not finished.

## Experiment parameters

One parameter changes across the grid. Schedulers stay fixed from Experiment 001.

| Parameter | Tested values | Purpose |
| --- | --- | --- |
| Initial architecture | ten starters (see table below) | Measures growth under the same search/LR settings |
| Random seed | `100`, `101`, `102`, `103` | Matched seeds; the script now plans four seeds per architecture |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Slope threshold | `3°` | Kept from Experiment 001 |
| LR warmup | logistic | Kept from Experiments 000/001 |
| Warmup length | `10` | Same as Experiment 001 |
| Dataset | MNIST | Classification task |
| Planned runs | `40` | `10` architectures × `4` seeds |
| Loaded runs so far | `12` | Six architectures present on disk |
| Completed runs so far | `10` | Charts below use completed seeds only |
| Generations | `10` | Ten training and architecture-decision cycles |
| Epochs per generation | `10` | Exactly `10` recorded epochs per generation |
| Target LR | `0.01` | Same as Experiment 001 |
| Batch size | `64` | Same as Experiment 001 |
| Simulation time limit | `500 s` | MCTS time budget |
| Simulation training epochs | `15` | Training budget inside simulation |
| Simulation set size | `2000` | Samples used by simulation |

### Starting graphs

| Name | Initial layers / pooling | Starting parameters | Why it is in the grid |
| --- | --- | ---: | --- |
| `big` | `conv1`, `conv2`, `linear`, `linear2` | `420` | Experiment 000/001 baseline |
| `medium` | `conv1`, `linear`, `linear2` | `276` | Needs sequential conv to rebuild the second stem conv |
| `very_small` | `conv1`, `linear2` | `76` | Needs sequential conv, then linear growth |
| `medium_h4` | medium depth, hidden size `4` | `96` | Weaker first residual / linear attachment |
| `medium_ch2_h8` | medium depth, channels `2`, hidden `8` | `122` | Width versus depth |
| `big_ch2_h8` | big depth, channels `2`, hidden `8` | `158` | Thin big stem |
| `very_small_ch2` | very small, channels `2` | not started | Thinner single-conv starter |
| `medium_max_pool_only` | medium depth, max pool only | not started | Pooling comparison |
| `medium_avg_pool_only` | medium depth, adaptive avg only | not started | Pooling comparison |
| `medium_no_pool` | medium depth, no pool | not started | Control without Rule-B seq-conv site |

### Progress of the grid

| Architecture | Seed `100` | Seed `101` | Seeds `102`–`103` |
| --- | --- | --- | --- |
| `big` | completed | completed | not started |
| `medium` | completed | completed | not started |
| `very_small` | completed | completed | not started |
| `medium_h4` | completed | completed | not started |
| `medium_ch2_h8` | completed | completed | not started |
| `big_ch2_h8` | running | running | not started |
| pooling / `very_small_ch2` | not started | not started | not started |

Current tables use the completed first-script architectures only. Re-run `python experiments/train_mnist_exp002_initial_architectures.py --board true` to skip completed seeds and continue. The script now also schedules seeds `102` and `103`, because two seeds are not enough for a stable ranking.

## Why this experiment exists

Experiment 001 left three linked problems:

1. Useful action gain was concentrated in the first action.
2. Very small finished near `49%` validation and could not take a natural sequential-convolution rebuild step.
3. Medium spent generations faking that rebuild with residual adds and deletes.

This experiment checks whether fixed schedulers plus legal sequential convolution produce healthier growth across many starters, and which starters are useful defaults for later work.

## Actions by training phase

The question is not only how many actions occur, but when they occur. Generations are grouped into three phases:

- early: generations `0–3`
- middle: generations `4–6`
- late: generations `7–9`

![Mean executed actions by training phase](/assets/experiments/002-actions-by-phase.png)

> [!CAPTION] Figure 1. Each phase has one bar per completed architecture. Values are mean action counts across completed seeds.

| Architecture | Completed seeds | Mean actions early `0–3` | Mean actions middle `4–6` | Mean actions late `7–9` | Mean total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `very_small` | `2` | `2.50` | `3.00` | `2.00` | `7.50` |
| `medium_h4` | `2` | `3.00` | `2.00` | `2.00` | `7.00` |
| `medium_ch2_h8` | `2` | `2.50` | `3.00` | `2.00` | `7.50` |
| `medium` | `2` | `2.00` | `3.00` | `1.50` | `6.50` |
| `big` | `2` | `1.50` | `2.50` | `2.00` | `6.00` |

Thinner starters act more in the early phase. Big waits longer. All completed starters still execute many middle and late actions. High late counts do not prove those late actions help. The recovery-window sections below test that.

## Accuracy gain after architecture actions

For each observable action, compare validation accuracy immediately before the action with validation accuracy at the end of the next generation. This gives the changed graph one generation to recover.

### Action order across completed runs

![Validation change by action order](/assets/experiments/002-action-order.png)

> [!CAPTION] Figure 2. Bars show the mean validation-accuracy change after the first, second, third, fourth, and fifth-or-later action. Dots show individual actions from all `10` completed runs. Values are percentage-point changes.

| Action order | Observed actions | Mean next-generation validation change |
| --- | ---: | ---: |
| First | `10` | `+17.80 percentage points` |
| Second | `10` | `+6.74 percentage points` |
| Third | `10` | `+2.33 percentage points` |
| Fourth | `10` | `+4.00 percentage points` |
| Fifth or later | `29` | `-0.12 percentage points` |

The first action is still the largest. The second through fourth actions stay positive on average, which is healthier than Experiment 001. From the fifth action onward, the mean is near zero or negative.

### Action order by architecture

![Validation change by action order and architecture](/assets/experiments/002-action-order-by-architecture.png)

> [!CAPTION] Figure 3. Each panel uses the same next-generation validation change as Figure 2. One panel is one completed architecture.

| Architecture | First-action mean | Second-action mean | Later-action pattern |
| --- | ---: | ---: | --- |
| `big` | `+25.19 percentage points` (`n=2`) | `-2.30 percentage points` (`n=2`) | First action still dominates |
| `medium` | `+40.57 percentage points` (`n=2`) | `+0.63 percentage points` (`n=2`) | Huge first residual-conv jump; late delete can destroy the run |
| `very_small` | `+12.33 percentage points` (`n=2`) | `+1.38 percentage points` (`n=2`) | First and third actions help; both first actions are sequential convolution |
| `medium_h4` | `+2.09 percentage points` (`n=2`) | `+4.80 percentage points` (`n=2`) | Early gains are spread; fourth action mean is about `+15.79` percentage points |
| `medium_ch2_h8` | `+8.85 percentage points` (`n=2`) | `+29.20 percentage points` (`n=2`) | First sequential linear, then residual convolution does the large climb |

`medium_h4` and `medium_ch2_h8` are the clearest multi-step starters so far. Plain `medium` still looks like Experiment 001: one large residual jump, then noisy later search.

### Action type

![Validation change by action type](/assets/experiments/002-action-types.png)

> [!CAPTION] Figure 4. Blue shows training-accuracy change and green shows validation-accuracy change over the next generation. Bars are means. Colored dots are individual actions.

| Action type | Observed actions | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Add residual convolution | `17` | `+17.78 percentage points` | `+15.34 percentage points` |
| Add residual linear | `2` | `+5.54 percentage points` | `+11.03 percentage points` |
| Add sequential convolution | `4` | `+6.60 percentage points` | `+6.30 percentage points` |
| Add sequential linear | `24` | `+1.67 percentage points` | `+2.73 percentage points` |
| Add sequential dropout | `12` | `-5.63 percentage points` | `+0.90 percentage points` |
| Delete layer | `10` | `-4.61 percentage points` | `-7.90 percentage points` |

Residual convolution remains strongest. Sequential convolution now appears and helps on average. Deletion remains harmful on average.

![Executed action counts by type and architecture](/assets/experiments/002-action-composition.png)

> [!CAPTION] Figure 5. Counts show what each starter actually executed across completed seeds, not only the mean accuracy effect.

Sequential convolution is present in the executed set, especially on `very_small`. Residual convolution and sequential linear still dominate the counts.

## Final results

### Result grouped by initial architecture

Architectures are ordered by starting parameter count.

![Mean final training and validation accuracy by architecture](/assets/experiments/002-final-accuracy-by-architecture.png)

> [!CAPTION] Figure 6. Each pair of bars is one architecture averaged across its completed seeds. Black dots are final validation for each seed. Architectures are ordered by starting parameters.

| Architecture | Completed seeds | Start params | Mean final training | Mean final validation | Mean peak validation | Mean actions | Mean final parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `very_small` | `2` | `76` | `53.41%` | `56.45%` | `57.22%` | `7.50` | `516` |
| `medium_h4` | `2` | `96` | `52.25%` | `57.05%` | `64.02%` | `7.00` | `412` |
| `medium_ch2_h8` | `2` | `122` | `65.94%` | `78.45%` | `82.05%` | `7.50` | `542` |
| `medium` | `2` | `276` | `52.03%` | `56.48%` | `85.89%` | `6.50` | `1372` |
| `big` | `2` | `420` | `82.55%` | `84.68%` | `85.38%` | `6.00` | `1182` |

Final means are lower than the matching Experiment 001 `3°` logistic cells for `big` and `medium`. Part of that is seed noise. With only two completed seeds, one bad path can move the mean a lot.

The important change versus Experiment 001 is the gap between medium and very small. In Experiment 001 under `3°`, medium finished near `81.84%` and very small near `51.50%`. Here their final means are almost the same (`56.48%` and `56.45%`). Very small rose a little because sequential convolution is legal. Medium fell a lot on the final mean because seed `100` collapsed. Peak validation still shows medium can reach the mid-`80%` range.

`medium_ch2_h8` is now the strongest completed thin starter on finals (`78.45%` mean). That supports keeping width-reduced medium variants in later experiments.

`big` remains best on average, but `84.68%` is also weaker than Experiment 001’s `92.06%` under the same schedulers. Seed `101` on big finished at only `75.18%`. More seeds are required before ranking these starters firmly.

### Comparison with Experiment 001 under the same schedulers

| Starter | Exp 001 `3°` mean final validation | Exp 002 mean final validation | Notes |
| --- | ---: | ---: | --- |
| `big` | `92.06%` | `84.68%` | Lower mean; large seed spread |
| `medium` | `81.84%` | `56.48%` | Final mean pulled down by one collapse |
| `very_small` | `51.50%` | `56.45%` | Higher; both seeds start with sequential convolution |

### Peak versus final validation

![Peak versus final validation](/assets/experiments/002-peak-vs-final.png)

> [!CAPTION] Figure 7. Purple is mean peak validation. Green is mean final validation. A large gap means late actions destroyed an earlier peak.

| Architecture | Seed | Peak validation | Final validation | Drop |
| --- | ---: | ---: | ---: | ---: |
| `medium` | `100` | `83.75%` | `29.15%` | `54.60 percentage points` |
| `medium_h4` | `100` | `63.48%` | `49.84%` | `13.64 percentage points` |
| `medium_ch2_h8` | `100` | `78.89%` | `72.33%` | `6.56 percentage points` |
| `medium` | `101` | `88.02%` | `83.81%` | `4.21 percentage points` |
| other completed seeds | — | — | — | under `2 percentage points` |

Peak versus final is the key warning chart. Medium seed `100` proves that a strong run can still be destroyed late. Until more seeds finish, peak validation and seed-level curves matter more than the medium final mean.

### Parameter growth

Architectures are ordered by starting parameter count: `very_small` (`76`), `medium_h4` (`96`), `medium_ch2_h8` (`122`), `medium` (`276`), `big` (`420`).

![Starting and final parameter counts](/assets/experiments/002-param-growth.png)

> [!CAPTION] Figure 8. Gray bars are starting parameters. Colored bars are mean final parameters. Dots are final counts per completed seed. Order follows starting size.

| Architecture | Start parameters | Mean final parameters | Mean growth |
| --- | ---: | ---: | ---: |
| `very_small` | `76` | `516` | `+440` |
| `medium_h4` | `96` | `412` | `+316` |
| `medium_ch2_h8` | `122` | `542` | `+420` |
| `medium` | `276` | `1372` | `+1096` |
| `big` | `420` | `1182` | `+762` |

Size alone does not decide accuracy. Medium can finish larger than big and still end weaker on the collapsed seed. Very small grows more than `medium_h4` on average, but both stay far below a strong MNIST solution.

## Final graph comparison

The board stores graphs as simplified PDFs. The images below are PNG renders. Originals are in `documentation/website/app/public/assets/experiments/exp002-graphs/`.

### Starting graphs before growth

![Very small starter](/assets/experiments/exp002-graphs/start-very_small-seed100.png)

> [!CAPTION] Figure 9. Very small starter: `conv1` then pooling into `linear2`. Starting parameters `76`.

![Medium h4 starter](/assets/experiments/exp002-graphs/start-medium_h4-seed101.png)

> [!CAPTION] Figure 10. `medium_h4` starter: medium depth with hidden size `4`. Starting parameters `96`.

![Medium starter](/assets/experiments/exp002-graphs/start-medium-seed101.png)

> [!CAPTION] Figure 11. Medium starter: `conv1`, `linear`, `linear2`. Starting parameters `276`.

![Big starter](/assets/experiments/exp002-graphs/start-big-seed100.png)

> [!CAPTION] Figure 12. Big starter: `conv1`, `conv2`, `linear`, `linear2`. Starting parameters `420`.

### Best completed final graph for each architecture

![Best very small final graph](/assets/experiments/exp002-graphs/final-very_small-seed100-val67.png)

> [!CAPTION] Figure 13. Best very small final graph: seed `100`, final validation `67.17%`. The first action was sequential convolution.

![Best medium_h4 final graph](/assets/experiments/exp002-graphs/final-medium_h4-seed101-val64.png)

> [!CAPTION] Figure 14. Best `medium_h4` final graph: seed `101`, final validation `64.26%`. Early sequential convolution appears on this seed.

![Best medium_ch2_h8 final graph](/assets/experiments/exp002-graphs/final-medium_ch2_h8-seed101-val85.png)

> [!CAPTION] Figure 15. Best completed `medium_ch2_h8` final graph: seed `101`, final validation `84.57%`.

![Best medium final graph](/assets/experiments/exp002-graphs/final-medium-seed101-val84.png)

> [!CAPTION] Figure 16. Best medium final graph: seed `101`, final validation `83.81%`. This is the healthy medium path.

![Best big final graph](/assets/experiments/exp002-graphs/final-big-seed100-val94.png)

> [!CAPTION] Figure 17. Best big final graph: seed `100`, final validation `94.18%`.

### Collapsed medium seed `100`

![Collapsed medium final graph](/assets/experiments/exp002-graphs/final-medium-seed100-collapsed.png)

> [!CAPTION] Figure 18. Medium seed `100` after the late collapse. Final validation `29.15%`. The residual convolution that carried most of the accuracy is gone.

This graph is the end of a strong run that failed late. Peak validation was `83.75%`. The last two executed actions were:

1. Generation `6`: `Add Seq Dropout Layer Action` between `linear` and `seq_linear_0` (`p=0.2`). Validation moved from `81.61%` to `77.09%` over the next generation. Training accuracy fell hard at the generation boundary.
2. Generation `8`: `Delete Layer Action` on `res_conv__0`. That residual convolution was the main early gain. After the delete, validation fell from `77.07%` to `29.15%`.

So the collapse is not mysterious. The run deleted the residual path that had rebuilt capacity, and left a thinner sequential linear/dropout chain.

## Training histories

![Training-accuracy curves by architecture](/assets/experiments/002-training-curves.png)

> [!CAPTION] Figure 19. Each panel is one architecture with completed seeds. Line color marks the seed.

Visible shapes:

1. `big` seed `100` rises into the high-accuracy region after the first residual convolution. Seed `101` stays lower after early dropout actions.
2. `medium` seed `101` rises and holds near the high-`70%` to mid-`80%` training region. Seed `100` rises, then collapses after the late delete of `res_conv__0`.
3. `very_small` seed `100` climbs in several steps after sequential convolution and later residual convolution. Seed `101` stays near `45%`.
4. `medium_h4` rises more slowly and stays below about `65%` peak validation.
5. `medium_ch2_h8` shows a clear two-step climb on both seeds: sequential linear, then residual convolution. Seed `101` finishes strongest among the thin medium variants.

The medium seed `100` curve is the reason the medium final mean looks almost identical to very small. Without that one collapse, medium would still look clearly stronger on finals.

## Seed effects and limitations

- Only `10` completed runs are available. Pooling starters and several width variants are unfinished. Seeds `102` and `103` are not started yet.
- Two seeds are not enough. The script now plans four seeds per architecture. Re-running the same command will fill the missing seeds without other script changes.
- Medium’s final mean is misleading until the collapsed seed is counted as a late-delete failure, not as a normal medium outcome.
- Sequential convolution helps early rebuild on `very_small`, but residual convolution still creates larger average gains.
- Stacked dropout and late deletion still appear.

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. The report keeps:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- graph PDF/PNG copies under `documentation/website/app/public/assets/experiments/exp002-graphs/`
- a normalized data snapshot at `documentation/website/data/experiments/experiment-002-initial-architectures.json`

`generate_experiment_002_charts.py` updates the snapshot when raw output exists. If raw output is missing, it reads the snapshot instead.

These documentation artifacts must be committed before the raw experiment folder is removed from this machine. After more seeds finish, re-run the chart script and refresh the tables on this page.

## Conclusions

1. Experiment 002 is an initial-architecture survey under fixed Experiment 001 schedulers. The scheduler pair is not the variable under test.
2. Sequential convolution is now used. Both completed `very_small` seeds take it as the first action.
3. Very small improves versus Experiment 001 (`51.50%` to `56.45%` mean final validation). Medium and very small finals are now almost equal, mainly because medium seed `100` collapsed after deleting `res_conv__0`.
4. Peak validation still separates medium from very small. Medium can reach the mid-`80%` range. The final mean cannot be trusted yet.
5. Early action gains are more multi-step than in Experiment 001. The first four action orders stay positive on average. Later actions remain risky.
6. `medium_h4` and `medium_ch2_h8` show the healthier early-gain shape. `medium_ch2_h8` currently has the strongest thin-starter finals (`78.45%` mean).
7. `big` remains strongest on average, but current means are lower and more seed-sensitive than Experiment 001. More seeds are required.
8. Late deletion after a high peak is unsafe. Medium seed `100` lost `54.60` percentage points after deleting the residual convolution that carried the earlier gain.

## Next steps

1. Finish the remaining architectures and the new seeds `102` and `103`, then regenerate charts and refresh this page.
2. Keep the Experiment 001 scheduler pair fixed while reading the finished architecture ranking.
3. Block or heavily penalize late deletion after a high validation peak.
4. Keep blocking stacked dropout on the same edge.
5. After the pooling starters finish, compare only max pool, only average pool, and no pool.
6. Prefer `medium_h4` or `medium_ch2_h8` when testing multi-step growth, because their early action-order curves are healthier than plain `medium`.
