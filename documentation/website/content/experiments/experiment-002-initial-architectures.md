# Experiment 002: Initial architectures

GrowingNN changes the network while it is training. Experiment 000 chose action-aware LR warmup. Experiment 001 chose `3°` slope scheduling with logistic warmup as the practical balance.

Experiment 001 also showed that the accuracy gain from architecture actions was not spread evenly across generations. Most useful gain came from the first action. Later actions added noise. The main structural cause was incomplete sequential-convolution insertion: thin starters could not rebuild a missing stem convolution in the natural sequential way, so search used residual workarounds, deletes, and stacked dropouts instead.

That insertion path is now fixed through `AddSeqConvLayer`. Experiment 002 keeps the Experiment 001 scheduler pair fixed and varies only the starting architecture. The goal is to see which initial graphs grow usefully for later experiments.

Raw output:

`experiments/output/train_mnist/runs/exp002_initial_architectures`

Script: `experiments/train_mnist_exp002_initial_architectures.py` — created 2026-08-04 14:38; revised after this report’s design findings.

Experiment runtime so far: oldest board start `2026-08-04T12:40:26Z`, newest update among available boards `2026-08-05T19:55:41Z`. Recorded training time across the loaded runs is about `53 hours`. The first grid is almost finished.

## Mistakes in initial architecture design

This section must be read before the result tables. The first grid mixed useful growth signals with bad stem designs, so architecture ranking is hard to trust as a final choice.

### Wrong starters: max pool and adaptive average pool stacked

Several starters put `max_pool2d` and then `adaptive_avg_pool2d` next to each other. That is a bad initial design. One stem should use one pooling style, not both in a row.

| Architecture | Start params | Pooling in the starter | Status |
| --- | ---: | --- | --- |
| `big` | `420` | `max_pool` → `max_pool` → `adaptive_avg_pool` | wrong |
| `big_ch2_h8` | `158` | same as `big` | wrong |
| `medium` | `276` | `max_pool` → `adaptive_avg_pool` | wrong |
| `medium_ch2_h8` | `122` | `max_pool` → `adaptive_avg_pool` | wrong |
| `medium_h4` | `96` | `max_pool` → `adaptive_avg_pool` | wrong |
| `very_small` | `76` | `max_pool` → `adaptive_avg_pool` | wrong |
| `very_small_ch2` | `38` | `max_pool` → `adaptive_avg_pool` | wrong |

### Starter that can stay from the first grid

| Architecture | Start params | Pooling in the starter | Status |
| --- | ---: | --- | --- |
| `medium_avg_pool_only` | `276` | only `adaptive_avg_pool` | valid compact starter |

So after checking the graphs, only one compact starter in this grid has a clean pooling design: `medium_avg_pool_only`.

### Completely neglected oversized controls

`medium_no_pool` and the old flatten-style `medium_max_pool_only` were also run, but they are removed from every table and chart on this page. Removing adaptive global pooling makes the first linear layer flatten a large spatial map, so they start at `50388` and `12756` parameters. That is a different capacity class. They are neglected completely in these results.

### Script update after these findings

The experiment script is already updated for the next grid. New runs write under `experiments/output/train_mnist/runs/exp002_initial_architectures_after_fix_1` so they stay separate from this first-grid folder. The revised grid compares topology only: same channels (`4`), same hidden size when a hidden linear exists (`16`), and the same pooling (`adaptive_avg_pool2d`) for every starter. Adaptive average pooling is used because it is the usual compact MNIST classification head and matched the only clean starter in this first grid.

| Name | Modules | Start params | Shared width |
| --- | --- | ---: | --- |
| `big` | `2×Conv2d` + `2×Linear` | `420` | channels `4`, hidden `16`, adaptive avg |
| `medium_1conv_2linear` | `1×Conv2d` + `2×Linear` | `276` | same |
| `medium_2conv_1linear` | `2×Conv2d` + `1×Linear` | `220` | same channels/pool |
| `small` | `1×Conv2d` + `1×Linear` | `76` | same channels/pool |

Order is largest start params first. `medium_1conv_2linear` is above `medium_2conv_1linear` because the hidden linear (`4→16` plus bias, then `16→10`) costs more than the second convolution (`4→4` kernels) plus a direct `4→10` classifier.

Revised grid constants: `GENERATIONS = 5`, `SIMULATION_TIME_SEC = 120`, seeds `100–103`. Width and pooling style are no longer experimental factors.

The tables and charts below still describe the first grid. They remain useful for growth-shape signals, but not as a clean architecture ranking.

## Experiment parameters

One parameter changes across the first grid. Schedulers stay fixed from Experiment 001.

| Parameter | Tested values | Purpose |
| --- | --- | --- |
| Initial architecture | eight compact starters below | Measures growth under the same search/LR settings |
| Random seed | `100`, `101`, `102`, `103` | Four matched seeds per architecture |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Slope threshold | `3°` | Kept from Experiment 001 |
| LR warmup | logistic | Kept from Experiments 000/001 |
| Warmup length | `10` | Same as Experiment 001 |
| Dataset | MNIST | Classification task |
| Compact starters in this report | `8` | Wrong oversized flatten controls are ignored |
| Generations | `10` | First grid used ten cycles; revised script uses `5` |
| Epochs per generation | `10` | Exactly `10` recorded epochs per generation |
| Target LR | `0.01` | Same as Experiment 001 |
| Batch size | `64` | Same as Experiment 001 |
| Simulation time limit | `500 s` | First grid MCTS budget; revised script uses `120 s` |
| Simulation training epochs | `15` | Training budget inside simulation |
| Simulation set size | `2000` | Samples used by simulation |

### Starting graphs

All lists and charts are ordered by starting parameter count, largest first.

| Name | Start params | Initial layers / pooling | Design note |
| --- | ---: | --- | --- |
| `big` | `420` | two convs + linear head; max pools then adaptive avg | wrong double pooling |
| `medium` | `276` | one conv + linear head; max then adaptive avg | wrong double pooling |
| `medium_avg_pool_only` | `276` | one conv + linear head; adaptive avg only | valid |
| `big_ch2_h8` | `158` | two thin convs + small head; same pooling as `big` | wrong double pooling; name is misleading |
| `medium_ch2_h8` | `122` | thin medium; max then adaptive avg | wrong double pooling |
| `medium_h4` | `96` | medium with hidden `4`; max then adaptive avg | wrong double pooling |
| `very_small` | `76` | one conv to logits; max then adaptive avg | wrong double pooling |
| `very_small_ch2` | `38` | thinner very small; max then adaptive avg | wrong double pooling |

### Progress of the first grid

Statuses below are read from each run’s `board/main.json` on disk.

| Architecture | Start params | Seed `100` | Seed `101` | Seed `102` | Seed `103` |
| --- | ---: | --- | --- | --- | --- |
| `big` | `420` | completed | completed | completed | completed |
| `medium` | `276` | completed | completed | completed | completed |
| `medium_avg_pool_only` | `276` | completed | completed | running | not started |
| `big_ch2_h8` | `158` | completed | completed | completed | completed |
| `medium_ch2_h8` | `122` | completed | completed | completed | completed |
| `medium_h4` | `96` | completed | completed | completed | completed |
| `very_small` | `76` | completed | completed | completed | completed |
| `very_small_ch2` | `38` | completed | completed | completed | completed |

## Why this experiment exists

Experiment 001 left three linked problems:

1. Useful action gain was concentrated in the first action.
2. Very small finished near `49%` validation and could not take a natural sequential-convolution rebuild step.
3. Medium spent generations faking that rebuild with residual adds and deletes.

After Experiment 001, sequential convolution was fixed so a convolution can be inserted before the flatten into a linear layer. Experiment 002 tests many initial architectures after that fix, under the same schedulers, to see which starters grow usefully.

## Actions by training phase

Generations are grouped into three phases:

- early: generations `0–3`
- middle: generations `4–6`
- late: generations `7–9`

![Mean executed actions by training phase](/assets/experiments/002-actions-by-phase.png)

> [!CAPTION] Figure 1. Each phase has one bar per architecture. Values are mean action counts across completed seeds. Leftmost architecture is the largest starter.

| Architecture | Start params | Seeds | Mean early `0–3` | Mean middle `4–6` | Mean late `7–9` | Mean total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `big` | `420` | `4` | `1.75` | `2.50` | `2.00` | `6.25` |
| `medium` | `276` | `4` | `2.00` | `3.00` | `1.75` | `6.75` |
| `medium_avg_pool_only` | `276` | `2` | `3.00` | `3.00` | `2.00` | `8.00` |
| `big_ch2_h8` | `158` | `4` | `2.25` | `2.75` | `1.75` | `6.75` |
| `medium_ch2_h8` | `122` | `4` | `2.75` | `2.75` | `2.00` | `7.50` |
| `medium_h4` | `96` | `4` | `3.00` | `2.50` | `2.00` | `7.50` |
| `very_small` | `76` | `4` | `2.75` | `3.00` | `2.00` | `7.75` |
| `very_small_ch2` | `38` | `4` | `3.50` | `3.00` | `2.00` | `8.50` |

Thinner starters act more in the early phase. Big waits longer. All compact starters also change structure in the middle and late phases. A large number of late actions does not mean those late actions were useful. Many late actions still hurt accuracy. Ten generations plus a `500 s` simulation budget made late noise hard to ignore; that is why the revised script cuts both.

## Accuracy gain after architecture actions

For each observable action, compare validation accuracy immediately before the action with validation accuracy at the end of the next generation.

### Action order across completed compact runs

![Validation change by action order](/assets/experiments/002-action-order.png)

> [!CAPTION] Figure 2. Bars show the mean validation-accuracy change after the first, second, third, fourth, and fifth-or-later action. Dots show individual actions. Values are percentage-point changes.

| Action order | Observed actions | Mean next-generation validation change |
| --- | ---: | ---: |
| First | `30` | `+20.43 percentage points` |
| Second | `30` | `+7.27 percentage points` |
| Third | `30` | `+1.54 percentage points` |
| Fourth | `30` | `+1.13 percentage points` |
| Fifth or later | `100` | `+0.94 percentage points` |

The first action is still the largest gain. The second action still helps on average. Later actions are small on average and often noisy.

### Action order by architecture

![Validation change by action order and architecture](/assets/experiments/002-action-order-by-architecture.png)

> [!CAPTION] Figure 3. Each panel uses the same next-generation validation change as Figure 2. Panels follow largest-to-smallest starting size.

| Architecture | Start params | First-action mean | Second-action mean | Later-action pattern |
| --- | ---: | ---: | ---: | --- |
| `big` | `420` | `+34.86 percentage points` (`n=4`) | `-1.31 percentage points` (`n=4`) | First residual-conv jump dominates |
| `medium` | `276` | `+28.62 percentage points` (`n=4`) | `-1.19 percentage points` (`n=4`) | Large first jump; finals remain unstable |
| `medium_avg_pool_only` | `276` | `+48.06 percentage points` (`n=2`) | `+4.82 percentage points` (`n=2`) | Very large first residual jump |
| `big_ch2_h8` | `158` | `+35.62 percentage points` (`n=4`) | `+0.34 percentage points` (`n=4`) | Same one-jump shape as `big` |
| `medium_ch2_h8` | `122` | `+7.74 percentage points` (`n=4`) | `+35.22 percentage points` (`n=4`) | Clearest multi-step ladder |
| `medium_h4` | `96` | `+4.47 percentage points` (`n=4`) | `+13.25 percentage points` (`n=4`) | Early gains spread across first actions |
| `very_small` | `76` | `+10.06 percentage points` (`n=4`) | `+2.35 percentage points` (`n=4`) | First action is sequential convolution |
| `very_small_ch2` | `38` | `+7.86 percentage points` (`n=4`) | `+3.43 percentage points` (`n=4`) | Sequential convolution early, still weak finals |

`medium_ch2_h8` and `medium_h4` show the best multi-step early gains. They still use the wrong double-pooling stem, so treat that result as a growth-shape signal, not as a final architecture choice.

### Action type

![Validation change by action type](/assets/experiments/002-action-types.png)

> [!CAPTION] Figure 4. Blue shows training-accuracy change and green shows validation-accuracy change over the next generation. Bars are means. Colored dots are individual actions.

| Action type | Observed actions | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Add residual convolution | `54` | `+16.44 percentage points` | `+15.26 percentage points` |
| Add sequential convolution | `11` | `+7.04 percentage points` | `+6.07 percentage points` |
| Add residual linear | `10` | `+2.55 percentage points` | `+4.18 percentage points` |
| Add sequential linear | `77` | `+1.63 percentage points` | `+2.49 percentage points` |
| Add sequential dropout | `35` | `-3.58 percentage points` | `-0.89 percentage points` |
| Delete layer | `33` | `-1.57 percentage points` | `-2.66 percentage points` |

Residual convolution remains strongest. Sequential convolution is used and helps on average. Dropout and deletion remain weak or harmful on average.

![Executed action counts by type and architecture](/assets/experiments/002-action-composition.png)

> [!CAPTION] Figure 5. Counts show what each starter actually executed across completed seeds.

## Final results

Order is largest starting parameters first. Mean charts include collapsed seeds. The best-seed chart below shows the optimistic envelope without those outliers.

![Mean final training and validation accuracy by architecture](/assets/experiments/002-final-accuracy-by-architecture.png)

> [!CAPTION] Figure 6. Mean final training and validation accuracy. Black dots are final validation for each seed.

| Architecture | Start params | Seeds | Mean final training | Mean final validation | Mean peak validation | Mean actions | Mean final parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `big` | `420` | `4` | `86.30%` | `85.58%` | `89.63%` | `6.25` | `1497` |
| `medium` | `276` | `4` | `57.44%` | `60.17%` | `75.91%` | `6.75` | `1428` |
| `medium_avg_pool_only` | `276` | `2` | `78.14%` | `84.30%` | `87.33%` | `8.00` | `2220` |
| `big_ch2_h8` | `158` | `4` | `83.47%` | `81.24%` | `87.05%` | `6.75` | `655` |
| `medium_ch2_h8` | `122` | `4` | `71.58%` | `78.43%` | `81.26%` | `7.50` | `568` |
| `medium_h4` | `96` | `4` | `57.15%` | `59.66%` | `64.35%` | `7.50` | `432` |
| `very_small` | `76` | `4` | `40.75%` | `46.01%` | `47.28%` | `7.75` | `427` |
| `very_small_ch2` | `38` | `4` | `31.04%` | `31.67%` | `33.45%` | `8.50` | `122` |

### Best-seed finals

This chart ignores bad outlier seeds. For each architecture it keeps only the best completed seed.

![Best-seed final training and validation accuracy](/assets/experiments/002-best-seed-accuracy-by-architecture.png)

> [!CAPTION] Figure 7. Best-seed final training and validation accuracy. Each bar is the strongest seed, not the mean.

| Architecture | Start params | Best seed | Best final training | Best final validation | Final parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `big` | `420` | `102` | `90.36%` | `94.67%` | `2068` |
| `medium` | `276` | `101` | `79.34%` | `83.81%` | `1924` |
| `medium_avg_pool_only` | `276` | `101` | `89.23%` | `91.44%` | `3108` |
| `big_ch2_h8` | `158` | `100` | `90.08%` | `88.68%` | `886` |
| `medium_ch2_h8` | `122` | `101` | `81.08%` | `84.57%` | `690` |
| `medium_h4` | `96` | `101` | `57.04%` | `64.26%` | `580` |
| `very_small` | `76` | `100` | `63.13%` | `67.17%` | `492` |
| `very_small_ch2` | `38` | `103` | `34.62%` | `35.60%` | `76` |

Why medium looks weak on the mean chart: two of four seeds went badly. Seed `100` collapsed after a late delete. Seed `102` stayed near `45%`. Seeds `101` and `103` reached the mid-`80%` range. So the mean is pulled down by bad paths, while the best-seed chart still shows a usable medium path. The deeper issue remains the wrong double-pooling stem.

`medium_avg_pool_only` is both cleaner in design and stronger on the seeds available so far (`84.30%` mean, `91.44%` best). Best-seed finals also show that stronger paths grow larger: valid medium ends at `3108` parameters, while the best thin multi-step path (`medium_ch2_h8`) ends at `690`.

### Comparison with Experiment 001 under the same schedulers

| Starter | Start params | Exp 001 `3°` mean final validation | Exp 002 four-seed mean final validation | Notes |
| --- | ---: | ---: | ---: | ---: |
| `big` | `420` | `92.06%` | `85.58%` | Lower mean; large seed spread |
| `medium` | `276` | `81.84%` | `60.17%` | Final mean hurt by collapses and weak seeds |
| `very_small` | `76` | `51.50%` | `46.01%` | Seq-conv appears, but four-seed mean falls |

### Parameter growth

![Starting and final parameter counts](/assets/experiments/002-param-growth.png)

> [!CAPTION] Figure 8. Gray bars are starting parameters. Colored bars are mean final parameters. Dots are final counts per completed seed.

| Architecture | Start parameters | Mean final parameters | Mean growth |
| --- | ---: | ---: | ---: |
| `big` | `420` | `1497` | `+1077` |
| `medium` | `276` | `1428` | `+1152` |
| `medium_avg_pool_only` | `276` | `2220` | `+1944` |
| `big_ch2_h8` | `158` | `655` | `+497` |
| `medium_ch2_h8` | `122` | `568` | `+446` |
| `medium_h4` | `96` | `432` | `+336` |
| `very_small` | `76` | `427` | `+351` |
| `very_small_ch2` | `38` | `122` | `+84` |

This table is the compact place to compare initial structure size. `big_ch2_h8` starts smaller than `medium`, so the “big” label is misleading.

## Training accuracy by generation

The question is whether longer runs still push training accuracy into a strong band, or whether later generations mostly rearrange weak models.

![Mean training accuracy by generation](/assets/experiments/002-train-acc-by-generation.png)

> [!CAPTION] Figure 9. Each bar group is one generation. Colors are architectures ordered by starting size. The dashed line marks `91%` training accuracy.

No compact starter reaches a mean end-of-generation training accuracy of `91%`. The strongest means stay in the mid-`80%` range late in training (`big` about `86%` at generation `9`). Several thin or double-pooling stems plateau far below that band. Ten generations therefore buy more late actions without delivering a clear strong-training finish. That supports cutting the revised grid to five generations.

## Starting graph comparison

Initial simplified graphs before growth. Order is largest starting parameters first. Captions mark the double-pooling mistake where it appears.

![big starter](/assets/experiments/exp002-graphs/start-big-seed100.png)

> [!CAPTION] Figure 10. `big` starter, `420` parameters. It uses max pooling and then adaptive average pooling. That stacked pooling is wrong.

![medium starter](/assets/experiments/exp002-graphs/start-medium-seed101.png)

> [!CAPTION] Figure 11. `medium` starter, `276` parameters. It also stacks max pooling and adaptive average pooling. That is wrong.

![medium_avg_pool_only starter](/assets/experiments/exp002-graphs/start-medium_avg_pool_only-seed101.png)

> [!CAPTION] Figure 12. `medium_avg_pool_only` starter, `276` parameters. Only adaptive average pooling. This is the valid compact medium design in this grid.

![big_ch2_h8 starter](/assets/experiments/exp002-graphs/start-big_ch2_h8-seed100.png)

> [!CAPTION] Figure 13. `big_ch2_h8` starter, `158` parameters. Same stacked pooling as `big`. The name “big” is misleading because it starts smaller than `medium`.

![medium_ch2_h8 starter](/assets/experiments/exp002-graphs/start-medium_ch2_h8-seed100.png)

> [!CAPTION] Figure 14. `medium_ch2_h8` starter, `122` parameters. Stacked max pool and adaptive average pool. Wrong design.

![medium_h4 starter](/assets/experiments/exp002-graphs/start-medium_h4-seed101.png)

> [!CAPTION] Figure 15. `medium_h4` starter, `96` parameters. Same stacked pooling mistake.

![very_small starter](/assets/experiments/exp002-graphs/start-very_small-seed100.png)

> [!CAPTION] Figure 16. `very_small` starter, `76` parameters. Max pool then adaptive average pool. Wrong design.

![very_small_ch2 starter](/assets/experiments/exp002-graphs/start-very_small_ch2-seed103.png)

> [!CAPTION] Figure 17. `very_small_ch2` starter, `38` parameters. Same stacked pooling mistake.

## Architecture comparisons

### Medium family versus big family

Compare the main depth cut: two-convolution stems versus one-convolution stems.

![big starter](/assets/experiments/exp002-graphs/start-big-seed100.png)

> [!CAPTION] Figure 18. Big family base: two convolutions before the linear head. Start params `420`.

![medium_avg_pool_only starter](/assets/experiments/exp002-graphs/start-medium_avg_pool_only-seed101.png)

> [!CAPTION] Figure 19. Valid medium family base: one convolution and one adaptive average pool. Start params `276`.

![Best big final graph](/assets/experiments/exp002-graphs/final-big-seed102-val95.png)

> [!CAPTION] Figure 20. Best big final graph: seed `102`, final validation `94.67%`.

![Best medium_avg_pool_only final graph](/assets/experiments/exp002-graphs/final-medium_avg_pool_only-seed101-val91.png)

> [!CAPTION] Figure 21. Best valid-medium final graph: `medium_avg_pool_only` seed `101`, final validation `91.44%`.

The old double-pooling `medium` should not be used as the medium side of this comparison. Its best seed still reaches `83.81%`, but the stem design is wrong.

![Collapsed medium final graph](/assets/experiments/exp002-graphs/final-medium-seed100-collapsed.png)

> [!CAPTION] Figure 22. Old double-pooling `medium` seed `100` after collapse. Final validation `29.15%`.

### Very small versus medium

![very_small starter](/assets/experiments/exp002-graphs/start-very_small-seed100.png)

> [!CAPTION] Figure 23. Very small starter. Start params `76`. Missing the hidden linear and using stacked pooling.

![medium_avg_pool_only starter](/assets/experiments/exp002-graphs/start-medium_avg_pool_only-seed101.png)

> [!CAPTION] Figure 24. Valid medium starter. Start params `276`. One hidden linear and one adaptive average pool.

![Best very small final graph](/assets/experiments/exp002-graphs/final-very_small-seed100-val67.png)

> [!CAPTION] Figure 25. Best very small final graph: seed `100`, final validation `67.17%`. Sequential convolution appears, but the run stays far below the valid medium.

Very small can take the sequential-convolution rebuild step. It still does not catch the valid medium starter. Also, the very-small stem itself still has the stacked pooling mistake, so a future very-small design should keep only one pooling style.

## Training histories

![Training-accuracy curves by architecture](/assets/experiments/002-training-curves.png)

> [!CAPTION] Figure 26. Each panel is one architecture, ordered largest starting size first. Line color marks the seed.

Visible shapes:

1. `big` splits into strong seeds (`100`, `102`) and weaker seeds (`101`, `103`).
2. Old `medium` has one collapse (`100`) and one weak path (`102`). That is why the mean looks bad.
3. `medium_avg_pool_only` stays high on the seeds completed so far.
4. `medium_ch2_h8` climbs in two steps and stays useful, but its stem still has stacked pooling.
5. `very_small` and `very_small_ch2` stay low.

## Seed effects and limitations

- Most starters in this grid use a wrong stacked pooling design. Only `medium_avg_pool_only` is clean.
- Old oversized flatten controls are ignored completely on this page.
- Medium means are sensitive to late deletes. Use the best-seed chart together with the mean chart.
- Sequential convolution helps thin rebuild, but it does not fix a bad pooling stem.
- Ten generations and a long simulation budget make late actions hard to interpret. Architecture comparisons are therefore noisy.

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. The report keeps:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- graph PDF/PNG copies under `documentation/website/app/public/assets/experiments/exp002-graphs/`
- a normalized data snapshot at `documentation/website/data/experiments/experiment-002-initial-architectures.json`

`generate_experiment_002_charts.py` ignores oversized flatten controls by starting parameter count. Ranking charts use the remaining compact starters only.

## Conclusions

1. Most starters in this first grid are wrong because they stack max pooling and adaptive average pooling. That makes clean ranking hard.
2. Only `medium_avg_pool_only` is a valid compact starter design in this experiment.
3. Oversized flatten controls are neglected completely because they are a different capacity class.
4. After the sequential-convolution fix, thin starters can insert convolution before flatten. That is visible on `very_small`.
5. Mean medium results look weak because of bad seeds. Best-seed medium still reaches `83.81%`, and valid `medium_avg_pool_only` reaches `91.44%` with `3108` final parameters.
6. `medium_ch2_h8` still shows the best multi-step early-gain shape, but it should not be kept as-is until its pooling stem is fixed.
7. `big_ch2_h8` should not be named as a “big” model. It starts with fewer parameters than `medium`.
8. No compact starter reaches a mean generation training accuracy of `91%`. Later generations mostly add noise. The revised script therefore uses `5` generations and `120 s` simulation time.

## Next steps

1. Run the revised Exp 002 grid from `experiments/train_mnist_exp002_initial_architectures.py` into `exp002_initial_architectures_after_fix_1`.
2. Regenerate charts after the new runs land under that after-fix output root.
3. Keep blocking late deletion after a high validation peak.
4. Keep blocking stacked dropout on the same edge.
