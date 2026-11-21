# Experiment 011: simulation-set selectors

Experiment 011 asks which rule for building the simulation set gives higher final validation accuracy. The search settings stay at the Experiment 010 Pass 2 winners. CIFAR-10 has a finished five-rule grid. MNIST has finished runs for stratified random, EL2N, and CCS, plus moderate seed 1. CIFAR-100 has not been run. MNIST moderate seed 2 was in generation 1 when this page was regenerated, so it is not in the means.

## Fixed settings

| Parameter | Value |
| --- | --- |
| Datasets in this page | CIFAR-10, finished; MNIST, seven of eight runs finished |
| Starter | one 8-channel convolution, ReLU, adaptive average pooling, then a linear head |
| CIFAR-10 starter | `3→8` convolution, `8→10` linear layer, `296` parameters |
| MNIST starter | `1→8` convolution, `8→10` linear layer, `152` parameters |
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
| Simulation share | `10%` of the clean training split |
| CIFAR-10 simulation images | `4,500` train, `1,120` validation |
| MNIST simulation images | `5,000` train, `1,250` validation |
| Selection metric | mean final validation accuracy |

The live model trains on the augmented full training set. Search trains and scores copies on the selected clean images. Official test accuracy is recorded and is not used to rank selectors.

## Selectors

Each selector draws the simulation set once, before generation 0, and reuses that same set for every search in the run. EL2N and CCS first fit a linear probe on the flattened pixels. Moderate uses Euclidean distance in that same pixel space. The probe is not the network that is being grown.

| Class | Where it was run | Idea |
| --- | --- | --- |
| `StratifiedRandomSimulationSet` | CIFAR-10 and MNIST | The current rule. Shuffle each class and keep an equal count. No example is preferred for being easy, hard, or typical. This is the control. |
| `El2nSimulationSet` | CIFAR-10 and MNIST | Train the linear probe for 10 epochs. Score each image by its average squared prediction error. Within each class, keep the images the probe is still getting wrong. |
| `CcsSimulationSet` | CIFAR-10 and MNIST | Use the same error scores. Drop the hardest 30% of each class, then spread the remaining budget across 50 difficulty bands so easy and mid-range images stay in the set. |
| `ModerateSimulationSet` | CIFAR-10, and MNIST seed 1 | For each class, find the mean pixel vector. Keep the images whose distance to that mean is closest to the median distance. The nearest prototypes and the farthest outliers are left out. |
| `ModerateCoverageSimulationSet` | CIFAR-10 only | Start from the central 30% distance band, then use k-center so the kept images are spread through that band instead of piled on one radius. Removed from the runner after the CIFAR-10 result. |

## Script and timeline

Script: `experiments/train_cifar10_exp011_simulation_sets.py`. Charts: `documentation/website/scripts/generate_experiment_011_charts.py`. Snapshot: `documentation/website/data/experiments/experiment-011-simulation-sets.json`.

CIFAR-10 histories were written from 19 November 2026 at 22:49 to 20 November 2026 at 19:41. Finished MNIST histories were written from 20 November 2026 at 23:06 to 21 November 2026 at 12:39. These are filesystem times. The snapshot has 17 runs. MNIST moderate seed 2 and CIFAR-100 are absent. Commit the charts and the snapshot before the raw run folder is deleted.

## Final accuracy

The chart shows the space of finished validation results for each selector. A long line means the runs landed far apart. A short line means they landed close together.

![Range of final validation accuracy](/assets/experiments/011-final-accuracy.svg)

> [!CAPTION] Figure 1. Final validation accuracy in percent. The gray line runs from the lowest finished result to the highest. The black point is the mean of those results. Circles are CIFAR-10 runs and diamonds are MNIST runs. Blue is seed 1 and orange is seed 2.

Stratified random has the longest line, from `38.60%` to `93.37%`. CCS also stretches from a CIFAR-10 run at `46.14%` up to two MNIST runs near `96%`, and its mean is the highest because those MNIST runs sit at the top. Moderate is shorter: its three finished runs lie between `57.90%` and `74.38%`. Moderate coverage is the shortest and the lowest, because it has only the two CIFAR-10 runs, `38.70%` and `46.78%`. The line mixes the two datasets, so a high MNIST point and a low CIFAR-10 point lengthen the same selector.

The table is final validation accuracy, in percent. The mean is the average of the finished cells in that row. MNIST moderate seed 2 has no cell because that run has not finished.

| Selector | CIFAR-10 seed 1 | CIFAR-10 seed 2 | MNIST seed 1 | MNIST seed 2 | Mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `CcsSimulationSet` | 61.44% | 46.14% | 96.43% | 96.76% | **75.19%** |
| `StratifiedRandomSimulationSet` | 38.60% | 61.06% | 93.37% | 84.89% | 69.48% |
| `El2nSimulationSet` | 54.56% | 56.66% | 74.17% | 81.80% | 66.80% |
| `ModerateSimulationSet` | 57.90% | 58.80% | 74.38% | — | 63.69% |
| `ModerateCoverageSimulationSet` | 38.70% | 46.78% | — | — | 42.74% |

Across the finished runs, `CcsSimulationSet` has the highest pooled mean validation, `75.19%`, and the highest pooled mean test, `75.26%`. `StratifiedRandomSimulationSet` is next at `69.48%` validation. `El2nSimulationSet` is `66.80%`, `ModerateSimulationSet` is `63.69%`, and `ModerateCoverageSimulationSet` is `42.74%`. This pool mixes the lower CIFAR-10 scale with the higher MNIST scale, so the dataset sections below are the comparison within one task.

### CIFAR-10

On CIFAR-10, validation order and official-test order are the same. `ModerateSimulationSet` has the highest mean and the smallest seed range, `0.90` percentage points. `CcsSimulationSet` seed 1 is the highest single CIFAR-10 run, `61.44%` validation, and seed 2 is `46.14%`. `StratifiedRandomSimulationSet` splits the same way: `61.06%` and `38.60%`.

![CIFAR-10 final accuracy](/assets/experiments/011-final-accuracy-cifar10.svg)

> [!CAPTION] Figure 2. CIFAR-10 only. Bars are means of seeds 1 and 2. Circles are validation seeds and diamonds are official-test seeds. Blue is seed 1 and orange is seed 2. Accuracy is in percent.

| Selector | Mean validation | Seed validation | Mean test | Mean train |
| --- | ---: | --- | ---: | ---: |
| `ModerateSimulationSet` | **58.35%** | `57.90–58.80%` | **59.07%** | 57.06% |
| `El2nSimulationSet` | 55.61% | `54.56–56.66%` | 55.68% | 48.02% |
| `CcsSimulationSet` | 53.79% | `46.14–61.44%` | 53.85% | 52.69% |
| `StratifiedRandomSimulationSet` | 49.83% | `38.60–61.06%` | 49.57% | 47.93% |
| `ModerateCoverageSimulationSet` | 42.74% | `38.70–46.78%` | 42.87% | 39.14% |

### MNIST

On MNIST, validation order and official-test order are again the same. `CcsSimulationSet` is highest, and its two seeds differ by `0.33` percentage points. `StratifiedRandomSimulationSet` is second. Both of its seeds finish above `84%` validation. `El2nSimulationSet` and the one finished moderate seed finish near `74–82%`.

![MNIST final accuracy](/assets/experiments/011-final-accuracy-mnist.svg)

> [!CAPTION] Figure 3. MNIST only. Bars are means of the finished seeds. Circles are validation seeds and diamonds are official-test seeds. Blue is seed 1 and orange is seed 2. Accuracy is in percent. Moderate shows seed 1 only.

| Selector | Finished seeds | Mean validation | Seed validation | Mean test | Mean train |
| --- | ---: | ---: | --- | ---: | ---: |
| `CcsSimulationSet` | 2 | **96.60%** | `96.43–96.76%` | **96.66%** | 86.99% |
| `StratifiedRandomSimulationSet` | 2 | 89.13% | `84.89–93.37%` | 89.76% | 75.74% |
| `El2nSimulationSet` | 2 | 77.99% | `74.17–81.80%` | 78.20% | 64.36% |
| `ModerateSimulationSet` | 1 | 74.38% | seed 1 only | 74.58% | 57.50% |

The dataset changes the ranking. Moderate leads the finished CIFAR-10 means. CCS leads the finished MNIST means, by `7.47` percentage points over stratified random.

## Training histories

The curves show when the seeds separate and whether a low run is low from the start.

![Training and validation histories](/assets/experiments/011-training-histories.svg)

> [!CAPTION] Figure 4. One row per dataset and one panel per finished selector. Solid lines are training accuracy and dashed lines are validation accuracy, in percent. Blue is seed 1 and orange is seed 2. Seed 2 also has a circle every 10 epochs. Dotted vertical lines are generation boundaries. CIFAR-10 moderate draws both seeds; they stay within about two percentage points, so the lines sit on top of each other. MNIST moderate shows seed 1 only.

![Mean training accuracy](/assets/experiments/011-mean-training-accuracy.svg)

> [!CAPTION] Figure 5. One line per selector. The line is the mean training accuracy, in percent, of every finished CIFAR-10 and MNIST run of that selector. Dotted vertical lines are generation boundaries.

![Mean validation accuracy](/assets/experiments/011-mean-validation-accuracy.svg)

> [!CAPTION] Figure 6. One line per selector. The line is the mean validation accuracy, in percent, of every finished CIFAR-10 and MNIST run of that selector. Dotted vertical lines are generation boundaries.

At the end of these pooled curves, CCS is above the other selectors. Stratified random is next. Moderate coverage, which has only the two CIFAR-10 runs, ends below the selectors that also have MNIST runs.

On CIFAR-10, the two moderate seed curves rise together and stay within about two percentage points after generation 2. Stratified random separates in generation 1: seed 2 is at `45.70%` validation by epoch 40, and seed 1 stays between about `31%` and `39%`. CCS separates after generation 3. EL2N seed 2 drops when convolutions are inserted at generations 2, 3, and 4, then later generations end above `44%` validation. Moderate coverage seed 1 stays near `33–39%`. Every CIFAR-10 run ends within `1` percentage point of its own best validation.

On MNIST, both CCS curves reach the mid-90s. Seed 1 is already at `89.31%` validation by epoch 60. Seed 2 dips to `79.49%` at epoch 80 and then joins seed 1 above `95%`. Stratified seed 1 jumps to `89.13%` by epoch 40 and ends at `93.37%`. Stratified seed 2 stays near `50%` through epoch 80 and ends at `84.89%`. EL2N seed 1 falls to `39.17%` at epoch 40 and climbs slowly to `74.17%`. EL2N seed 2 stays near `48–57%` until the last generation, then ends at `81.80%`. The one moderate seed climbs from about `48%` at epoch 60 to `74.38%` at epoch 200.

On MNIST, training accuracy stays below validation accuracy for every finished selector. The gap is about `10` percentage points for CCS and about `17` percentage points for moderate seed 1 at the end. The curves still rank the selectors in the same order as the final validation table.

## Limitations

- MNIST `ModerateSimulationSet` has one finished seed. Its `74.38%` validation is not a two-seed mean.
- CIFAR-100 has no runs. A selector that leads on MNIST and not on CIFAR-10 can still change again on a 100-class task.
- CIFAR-10 `CcsSimulationSet` and `StratifiedRandomSimulationSet` have seed ranges of `15.30` and `22.46` percentage points. Their means are sensitive to which seed is included.
- EL2N, CCS, and moderate score examples from flattened pixels, not from the growing convolutional network.
- `ModerateCoverageSimulationSet` remains in the CIFAR-10 table because those two runs finished. The class is no longer in the runner.
- Timeline dates are filesystem modification times of `history.pt`.

## Conclusions

1. On finished CIFAR-10 runs, `ModerateSimulationSet` has the highest mean validation, `58.35%`, and the highest mean official test, `59.07%`. Both seeds finish between `57.90%` and `58.80%` validation.
2. On finished MNIST runs, `CcsSimulationSet` has the highest mean validation, `96.60%`, and the highest mean official test, `96.66%`. Both seeds finish between `96.43%` and `96.76%` validation.
3. The ranking does not carry from one dataset to the other. Moderate leads CIFAR-10. CCS leads MNIST. The one finished MNIST moderate seed, `74.38%` validation, is in the lower MNIST group.
4. `StratifiedRandomSimulationSet` can finish near the best run on CIFAR-10, at `61.06%`, and can also stay at `38.60%`. On MNIST both seeds finish above `84%`, with a gap of `8.48` percentage points.
5. `ModerateCoverageSimulationSet` has the lowest CIFAR-10 mean validation, `42.74%`. That is why it was removed from the runner.
6. Official test accuracy follows the validation ranking on both datasets. The search images and the official test images agree on the order of these finished means.

## Next experiments

1. Let MNIST `ModerateSimulationSet` seed 2 finish, then put that seed into this table.
2. Run CIFAR-100 only if there is time. The present disagreement between CIFAR-10 and MNIST is already enough to avoid treating moderate as the default for every dataset.
3. Recompute EL2N, CCS, and moderate scores from the starter convolutional network instead of flattened pixels.
4. On one generation, score the same legal mutations on the selected subset and on the full clean validation split, and compare the order of those scores.
