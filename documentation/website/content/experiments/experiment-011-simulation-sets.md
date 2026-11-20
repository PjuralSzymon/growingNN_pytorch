# Experiment 011: simulation-set selectors

Experiment 011 compares five ways to build the images used inside Monte Carlo search. The search settings stay at the Experiment 010 Pass 2 winners. Every arm uses 10% of the clean CIFAR-10 training split. The question is which selection rule has the higher final validation accuracy, and whether the two seeds agree.

## Fixed settings

| Parameter | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Starter | one `3→8` convolution, ReLU, adaptive average pooling, `8→10` linear layer (`296` parameters) |
| Generations × epochs | `10` × `20` (`200` epochs) |
| Learning rate | cosine from `0.1` to `0.001` |
| Seeds | `1`, `2` |
| Search | `montecarlo_alg`, always on except the last generation |
| UCB1 `C` | `1.0` |
| Square-root UCB1 | off |
| Rollout value propagation | on |
| Rollout depth | `1` |
| Rollout epochs | `1` |
| Rollout learning rate | `0.0001`, applied at the `0.001` floor |
| Minimum search iterations | `3` |
| Search time | `32 s` |
| Simulation training images | `4,500` (`10%` of `45,000`) |
| Simulation validation images | `1,120` |
| Selection metric | mean final validation accuracy |

The live model trains on the augmented full training set. Search trains and scores copies on the selected clean images. The official test set is recorded after each generation and is not used to rank selectors.

## Selectors

Each selector is one class. It draws the simulation set once, before generation 0, and that set is reused for every search in the run.

| Order | Class | Rule |
| --- | --- | --- |
| 1 | `StratifiedRandomSimulationSet` | Current rule. An equal random count from every class. |
| 2 | `El2nSimulationSet` | A linear probe trains for 10 epochs on flattened images. Each class keeps the examples with the largest average squared prediction error. |
| 3 | `CcsSimulationSet` | The same probe scores. The hardest 30% of each class is dropped, then the budget is spread across 50 difficulty bands. |
| 4 | `ModerateSimulationSet` | Each class keeps the examples whose flattened-pixel distance to the class mean is closest to the median distance. |
| 5 | `ModerateCoverageSimulationSet` | The central 30% distance band is kept, then k-center spreads the budget inside that band. |

## Script and timeline

Script: `experiments/train_cifar10_exp011_simulation_sets.py`

Charts: `documentation/website/scripts/generate_experiment_011_charts.py`

Run output: `experiments/output/train_cifar10/runs/exp011_simulation_sets`

Snapshot: `documentation/website/data/experiments/experiment-011-simulation-sets.json`

The ten `history.pt` files were written from 19 November 2026 at 22:49 to 20 November 2026 at 19:41, local filesystem time. Those timestamps are file modification times, not a recorded experiment clock. All `10` of `10` runs finished `200` epochs.

The chart files and the JSON snapshot are the published copy of these results. Commit them before the raw run folder is deleted.

## Final accuracy

Final validation accuracy is the metric used to rank selectors. Official test accuracy checks whether that ranking agrees with images the search never saw.

![Final validation and test accuracy](/assets/experiments/011-final-accuracy.svg)

> [!CAPTION] Figure 1. Bars are the mean of seeds 1 and 2. Circles are validation seeds and diamonds are official-test seeds. Blue is seed 1 and orange is seed 2. Accuracy is in percent.

| Selector | Mean validation | Seed validation | Mean test | Mean train |
| --- | ---: | --- | ---: | ---: |
| `ModerateSimulationSet` | **58.35%** | `57.90–58.80%` | **59.07%** | 57.06% |
| `El2nSimulationSet` | 55.61% | `54.56–56.66%` | 55.68% | 48.02% |
| `CcsSimulationSet` | 53.79% | `46.14–61.44%` | 53.85% | 52.69% |
| `StratifiedRandomSimulationSet` | 49.83% | `38.60–61.06%` | 49.57% | 47.93% |
| `ModerateCoverageSimulationSet` | 42.74% | `38.70–46.78%` | 42.87% | 39.14% |

`ModerateSimulationSet` has the highest mean validation and the highest mean official test. The validation order and the test order are the same. Its two seeds differ by `0.90` percentage points. `El2nSimulationSet` is `2.74` percentage points lower on mean validation, with a seed range of `2.10` percentage points.

The other means hide a split inside the pair. `CcsSimulationSet` seed 1 finishes at `61.44%` validation, the highest single run, and seed 2 finishes at `46.14%`. `StratifiedRandomSimulationSet` seed 2 finishes at `61.06%`, and seed 1 finishes at `38.60%`.

| Selector | Seed | Final train | Final validation | Best validation | Final test | Final parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `StratifiedRandomSimulationSet` | 1 | 36.98% | 38.60% | 39.32% | 38.66% | `672` |
| `StratifiedRandomSimulationSet` | 2 | 58.88% | 61.06% | 61.56% | 60.47% | `5,000` |
| `El2nSimulationSet` | 1 | 45.14% | 54.56% | 54.60% | 54.74% | `3,672` |
| `El2nSimulationSet` | 2 | 50.90% | 56.66% | 56.80% | 56.62% | `3,264` |
| `CcsSimulationSet` | 1 | 59.58% | 61.44% | 61.78% | 61.10% | `5,392` |
| `CcsSimulationSet` | 2 | 45.81% | 46.14% | 46.48% | 46.60% | `2,504` |
| `ModerateSimulationSet` | 1 | 56.57% | 57.90% | 58.30% | 58.67% | `5,000` |
| `ModerateSimulationSet` | 2 | 57.54% | 58.80% | 58.84% | 59.46% | `5,000` |
| `ModerateCoverageSimulationSet` | 1 | 37.40% | 38.70% | 39.18% | 38.78% | `2,372` |
| `ModerateCoverageSimulationSet` | 2 | 40.88% | 46.78% | 47.46% | 46.96% | `4,456` |

## Training histories

The curves show when the seeds separate and whether a run recovers after a mutation.

![Training and validation histories](/assets/experiments/011-training-histories.svg)

> [!CAPTION] Figure 2. One panel per selector. Solid lines are training accuracy and dashed lines are validation accuracy, in percent. Blue is seed 1 and orange is seed 2. Dotted vertical lines are generation boundaries, every 20 epochs.

The two `ModerateSimulationSet` curves rise together. By the end of generation 2 both validation curves are near `49–50%`, and they stay within about two percentage points of each other through epoch 200. Training and validation stay close for both seeds.

`StratifiedRandomSimulationSet` separates in generation 1. Seed 2 validation is `45.70%` at epoch 40 and keeps climbing. Seed 1 stays between about `31%` and `39%` for the remaining epochs. `CcsSimulationSet` separates later: after generation 3, seed 1 continues upward and seed 2 remains in the mid-30s to mid-40s.

`El2nSimulationSet` seed 2 drops at the convolution inserted at the ends of generations 2, 3, and 4. The recorded validation immediately after those actions is `38.96%`, `30.32%`, and `28.78%`. Each of those generations is followed by a later generation whose end validation is back above `44%`. Seed 1 training stays about `9` percentage points below its validation curve at the end. The other selectors finish with a smaller train–validation gap.

`ModerateCoverageSimulationSet` seed 1 is almost flat, moving from `33.18%` validation at epoch 20 to `38.70%` at epoch 200. Seed 2 falls to `29.10%` at epoch 40 and then climbs slowly, ending at `46.78%`.

Every run ends within `1` percentage point of its own best validation. The low runs were low across the 200 epochs. They are not late collapses of an earlier high peak.

The two runs below `40%` final validation, stratified seed 1 and moderate-coverage seed 1, executed no convolution. Their actions were linear layers, neuron edits, and dropout. Both `ModerateSimulationSet` runs executed eight convolution actions and finished at `5,000` parameters. This is the action record of these ten searches. It is not a separate test that convolution actions are required.

## Limitations

- Each mean uses two seeds. `CcsSimulationSet` and `StratifiedRandomSimulationSet` have seed ranges of `15.30` and `22.46` percentage points, so their means are sensitive to which seed is included.
- `El2nSimulationSet` and `CcsSimulationSet` score examples with a linear probe on flattened pixels. `ModerateSimulationSet` and `ModerateCoverageSimulationSet` measure distance in that same pixel space. The scores are not taken from the growing convolutional network.
- The budget is fixed at `4,500` training images. This page does not re-test the 5%, 30%, or 40% sizes from Experiment 010.
- Timeline dates are filesystem modification times of `history.pt`.

## Conclusions

1. `ModerateSimulationSet` has the highest mean final validation, `58.35%`, and the highest mean official test, `59.07%`. Both seeds finish between `57.90%` and `58.80%` validation.
2. `El2nSimulationSet` is second on both validation and test. Its seed range is `2.10` percentage points, and its mean validation is `2.74` percentage points below moderate.
3. `CcsSimulationSet` contains the best single run, seed 1 at `61.44%` validation, and a second seed at `46.14%`. The mean does not describe either seed well.
4. `StratifiedRandomSimulationSet` also splits: `61.06%` and `38.60%` validation. The current rule can reach the top of this grid and can also stay near `39%`.
5. `ModerateCoverageSimulationSet` has the lowest mean validation, `42.74%`, and the lowest mean test, `42.87%`.
6. Two seeds are enough to prefer `ModerateSimulationSet` over `ModerateCoverageSimulationSet` in this setup. They are not enough to retire `StratifiedRandomSimulationSet`, because one of its seeds matches the best run in the grid.

## Next experiments

1. Add seed `3` for `ModerateSimulationSet`, `El2nSimulationSet`, and `StratifiedRandomSimulationSet`.
2. Recompute the EL2N, CCS, and moderate scores from the starter convolutional network instead of flattened pixels.
3. On one generation, score the same legal mutations on the selected subset and on the full clean validation split, and compare the order of those scores.
