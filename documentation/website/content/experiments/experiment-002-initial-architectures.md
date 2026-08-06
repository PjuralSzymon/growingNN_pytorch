# Experiment 002: Initial architectures

GrowingNN changes the network while it is training. Experiment 000 chose action-aware LR warmup. Experiment 001 chose `3°` slope scheduling with logistic warmup. After sequential convolution could be inserted before flatten, this experiment asks which starting layout grows usefully when width and pooling stay fixed.

Script: `experiments/train_mnist_exp002_initial_architectures.py`

Published results:

`experiments/output/train_mnist/runs/exp002_initial_architectures_after_fix_1`

## Two runs

### First run: design mistakes

Folder `exp002_initial_architectures` mixed bad starters with the topology question: stacked pooling, oversized flatten controls, and width/pooling changing together. Those runs are history only.

### Corrected run: topology only

The corrected grid compares layouts only: shared `adaptive_avg_pool2d`, channels `4`, hidden size `16` when present, no adjacent pools, `5` generations, `120 s` simulation, seeds `100–103`. Runtime about `6.3 hours` for `16` completed runs.

## Experiment parameters

| Parameter | Tested values | Purpose |
| --- | --- | --- |
| Initial architecture | four topology starters | Growth under shared width and pooling |
| Random seed | `100`, `101`, `102`, `103` | Four matched seeds per architecture |

| Fixed parameter | Value |
| --- | ---: |
| Slope threshold | `3°` |
| LR warmup | logistic |
| Warmup length | `10` |
| Dataset | MNIST |
| Channels | `4` |
| Hidden linear size | `16` when present |
| Pooling | `adaptive_avg_pool2d` to `1×1` |
| Generations | `5` |
| Epochs per generation | `10` |
| Target LR | `0.01` |
| Batch size | `64` |
| Simulation time | `120 s` |

### Starting graphs

| Name | Modules | Start params |
| --- | --- | ---: |
| `big` | `2×Conv2d` + `2×Linear` | `420` |
| `medium_1conv_2linear` | `1×Conv2d` + `2×Linear` | `276` |
| `medium_2conv_1linear` | `2×Conv2d` + `1×Linear` | `220` |
| `small` | `1×Conv2d` + `1×Linear` | `76` |

Chart labels: `med 1c+2l` = `medium_1conv_2linear`, `med 2c+1l` = `medium_2conv_1linear`.

All `16` boards completed.

## Actions by generation

Counts are totals across all four seeds, not means.

![Executed action counts by generation](/assets/experiments/002-actions-by-generation.png)

> [!CAPTION] Figure 1. Action counts in generations `0–3`. Generation `4` had zero actions on every seed.

| Architecture | Gen `0` | Gen `1` | Gen `2` | Gen `3` | Gen `4` | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `big` | `4` | `2` | `4` | `4` | `0` | `14` |
| `medium_1conv_2linear` | `3` | `3` | `4` | `1` | `0` | `11` |
| `medium_2conv_1linear` | `4` | `4` | `3` | `3` | `0` | `14` |
| `small` | `4` | `3` | `4` | `3` | `0` | `14` |

`medium_1conv_2linear` acts least and later finishes strongest on the mean.

## Accuracy gain after architecture actions

For each action, compare end-of-generation training and validation accuracy with the end of the next generation.

### Action order

![Training and validation change by action order](/assets/experiments/002-action-order.png)

> [!CAPTION] Figure 2. Blue is training change. Green is validation change. Bars are means. Dots are individual actions.

| Action order | n | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| First | `16` | `+11.97 percentage points` | `+11.80 percentage points` |
| Second | `16` | `+6.77 percentage points` | `+5.88 percentage points` |
| Third | `15` | `+9.15 percentage points` | `+9.71 percentage points` |
| Fourth | `6` | `+3.74 percentage points` | `+5.78 percentage points` |

Validation gains look consistently positive. Training gains are also positive on average, but later actions can favor validation more than training. That matters because action choice is scored on validation.

### Action order by architecture

![Training and validation change by action order and architecture](/assets/experiments/002-action-order-by-architecture.png)

> [!CAPTION] Figure 3. Same recovery window as Figure 2, one panel per starter.

| Architecture | First train / val | Second train / val |
| --- | --- | --- |
| `big` | `+32.37` / `+32.80` | `+1.67` / `-3.33` |
| `medium_1conv_2linear` | `+5.41` / `+5.28` | `+15.72` / `+15.43` |
| `medium_2conv_1linear` | `+1.47` / `+1.93` | `+5.00` / `+6.83` |
| `small` | `+8.63` / `+7.22` | `+4.70` / `+4.59` |

Values are percentage points. `big` is strong only when the first action is residual convolution. `medium_1conv_2linear` shows the clearest multi-step climb.

### Action type

![Training and validation change by action type](/assets/experiments/002-action-types.png)

> [!CAPTION] Figure 4. Mean training and validation change by action type.

| Action type | n | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Add residual convolution | `10` | `+33.32 percentage points` | `+29.01 percentage points` |
| Add residual linear | `1` | `+7.15 percentage points` | `+8.14 percentage points` |
| Add sequential convolution | `12` | `+6.70 percentage points` | `+5.95 percentage points` |
| Add sequential linear | `13` | `+3.69 percentage points` | `+3.13 percentage points` |
| Add sequential dropout | `17` | `-0.54 percentage points` | `+3.12 percentage points` |

![Executed action counts by type and architecture](/assets/experiments/002-action-composition.png)

> [!CAPTION] Figure 5. Executed action counts by type and starter. Dropout is the most common action.

Dropout is the key imbalance: mean training change is slightly negative, mean validation change is still positive, and it is used too often. If simulation picks actions by validation score, dropout is favored even when it does not help learning. Stacked early dropout is the main collapse pattern on weak `big` and weak `medium_2conv_1linear` seeds.

## Training histories

![Training-accuracy curves by architecture](/assets/experiments/002-training-curves.png)

> [!CAPTION] Figure 6. Training curves by starter. Line color marks the seed.

### `big`

- Seeds `102` and `103`: first action is residual convolution. They reach about `82–84%` final validation.
- Seeds `100` and `101`: first actions are sequential dropout, then more dropout. They stay near `26%` validation.

### `medium_1conv_2linear`

Most stable useful starter. Strong seeds grow the hidden linear path, add residual convolution, then keep training. Best finals are about `83–84%` validation. It does not pass `90%` because after the residual-convolution jump the strong seeds stop acting; generation `4` is empty.

### `medium_2conv_1linear`

Worst starter. It begins with two convolutions and no hidden linear. Search often chooses dropout instead of building that missing head. Best seed only reaches `50.80%` validation.

### `small`

Every seed starts with sequential convolution, so the rebuild path from Experiment 001 now works. The best seed reaches `62.03%`. It grows in the right direction, but five generations are not enough to catch the better medium.

## Starting and final graphs

![big starter](/assets/experiments/exp002-graphs/start-big-seed103.png)

> [!CAPTION] Figure 7. `big` starter, `420` parameters.

![medium_1conv_2linear starter](/assets/experiments/exp002-graphs/start-medium_1conv_2linear-seed102.png)

> [!CAPTION] Figure 8. `medium_1conv_2linear` starter, `276` parameters.

![medium_2conv_1linear starter](/assets/experiments/exp002-graphs/start-medium_2conv_1linear-seed103.png)

> [!CAPTION] Figure 9. `medium_2conv_1linear` starter, `220` parameters.

![small starter](/assets/experiments/exp002-graphs/start-small-seed103.png)

> [!CAPTION] Figure 10. `small` starter, `76` parameters.

![Best big final graph](/assets/experiments/exp002-graphs/final-big-seed103-val84.png)

> [!CAPTION] Figure 11. Best big final: seed `103`, validation `83.74%`.

![Best medium_1conv_2linear final graph](/assets/experiments/exp002-graphs/final-medium_1conv_2linear-seed102-val84.png)

> [!CAPTION] Figure 12. Best `medium_1conv_2linear` final: seed `102`, validation `83.97%`. Adaptive average pool is still the original starter pool after `conv1`. This run added residual convolution and linear capacity, not a second sequential stem convolution.

![Best small final graph](/assets/experiments/exp002-graphs/final-small-seed103-val62.png)

> [!CAPTION] Figure 13. Best small final: seed `103`, validation `62.03%`.

![Weak big final graph](/assets/experiments/exp002-graphs/final-big-seed100-weak.png)

> [!CAPTION] Figure 14. Weak big seed `100`, validation `26.04%`. Three sequential dropouts before a late sequential convolution.

## Final results

All completed seeds, including collapsed big outliers.

![Mean final training and validation accuracy by architecture](/assets/experiments/002-final-accuracy-by-architecture.png)

> [!CAPTION] Figure 15. Mean finals with per-seed dots. Blue dots are training. Black dots are validation.

| Architecture | Mean final training | Mean final validation | Mean of seed peak validation | Mean actions | Mean final parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `big` | `48.47%` | `54.59%` | `56.80%` | `3.50` | `962` |
| `medium_1conv_2linear` | `70.51%` | `70.51%` | `71.84%` | `2.75` | `960` |
| `medium_2conv_1linear` | `30.88%` | `37.38%` | `38.20%` | `3.50` | `479` |
| `small` | `41.16%` | `46.41%` | `46.57%` | `3.50` | `360` |

“Mean of seed peak validation” is the mean, across seeds, of each seed’s highest validation during the run.

![Best-seed final training and validation accuracy](/assets/experiments/002-best-seed-accuracy-by-architecture.png)

> [!CAPTION] Figure 16. Best-seed finals only.

| Architecture | Best seed | Best final training | Best final validation | Final parameters |
| --- | ---: | ---: | ---: | ---: |
| `big` | `103` | `81.72%` | `83.74%` | `1304` |
| `medium_1conv_2linear` | `102` | `80.44%` | `83.97%` | `1140` |
| `medium_2conv_1linear` | `103` | `48.21%` | `50.80%` | `516` |
| `small` | `103` | `58.53%` | `62.03%` | `392` |

![Starting and final parameter counts](/assets/experiments/002-param-growth.png)

> [!CAPTION] Figure 17. Parameter counts for all seeds. This is parameter growth, not accuracy.

## Final results without big outliers

Big seeds `100` and `101` collapse after early stacked dropout. They pull big means down and make big look similar to medium on average size and accuracy. The charts below remove those two runs.

![Mean final accuracy without collapsed big seeds](/assets/experiments/002-final-accuracy-without-big-outliers.png)

> [!CAPTION] Figure 18. Mean finals after removing big seeds `100` and `101`.

| Architecture | Seeds kept | Mean final training | Mean final validation | Mean of seed peak validation | Mean final parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `big` | `102`, `103` | `75.71%` | `82.75%` | `86.77%` | `1294` |
| `medium_1conv_2linear` | `4` | `70.51%` | `70.51%` | `71.84%` | `960` |
| `medium_2conv_1linear` | `4` | `30.88%` | `37.38%` | `38.20%` | `479` |
| `small` | `4` | `41.16%` | `46.41%` | `46.57%` | `360` |

![Parameter growth without collapsed big seeds](/assets/experiments/002-param-growth-without-big-outliers.png)

> [!CAPTION] Figure 19. Parameter growth after removing big seeds `100` and `101`.

Without the collapsed big runs, successful big ends larger and stronger than medium on the mean. Best-seed `medium_1conv_2linear` still matches best-seed big near `84%`, with a cheaper start and more stable seeds.

## Conclusions

1. Experiment 001 could not make medium and small competitive with big. Adding sequential convolution before flatten opened that path. In this corrected grid, medium and small clearly grow farther than before.
2. The best corrected starter is `medium_1conv_2linear`: cheap start, stable seeds, and best-seed accuracy matching big.
3. `big` can match that band only when the first action is residual convolution. Early stacked dropout collapses it.
4. `medium_2conv_1linear` is a bad start here: two convolutions and no hidden linear, then too much early dropout, so growth stays blocked.
5. `small` can now rebuild through sequential convolution and move toward medium/big structure. In five generations it still finishes below the better medium, but the direction is open.
6. Dropout is used too often. It can raise validation while training stays flat or falls. Because action choice uses validation score, search is biased toward dropout. That is the main policy problem found here.
7. No starter stably passed `90%`. Strong medium seeds stop near `84%` after one residual-convolution jump and then take no generation-`4` action.

## Next steps

1. Use `medium_1conv_2linear` as the default MNIST starter.
2. Run a dedicated experiment on action scoring: compare choosing actions by validation accuracy, training accuracy, or a mix of both. The dropout bias makes this the next priority.
3. Penalize or block stacked sequential dropout early in a run.
4. Adjust simulation scoring so regularization-only gains on validation are not over-selected when training does not improve.
5. After the scoring fix, re-check whether the stable medium path can pass `90%` with the same short horizon.
