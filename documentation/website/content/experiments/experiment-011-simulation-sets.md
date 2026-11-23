# Experiment 011: simulation-set selectors

## Context

The search judges each candidate architecture by training a copy for a short time on a small simulation set and measuring its accuracy. Experiment 010 showed that the size of this set matters more than any other search setting: 10% of the training data was far better than 128 images. This experiment keeps the size at 10% and asks whether *which* images are chosen also matters. A random sample is the control; the other rules prefer hard, typical or well-spread images, ideas taken from data-pruning research.

## Summary

- **Question:** which images should the search train and score candidates on?
- **Best overall: CCS** (75.2% mean over all runs). **Best on CIFAR-10: Moderate** (58.4%); CCS leads MNIST (96.6%) (Table 3, Figures 1–3).
- **Seeds matter more than most rules:** stratified random ends at 38.6% or 61.1% on CIFAR-10 depending on the seed.
- **Moderate coverage was clearly worst** (42.7% on CIFAR-10) and was removed.

| Setting | Value |
| --- | --- |
| Datasets, seeds | CIFAR-10 and MNIST, seeds 1 and 2 (MNIST moderate: seed 1 only) |
| Starter | Conv →8, ReLU, pool, Linear 8→10 |
| Schedule | 10 generations × 20 epochs |
| Search | Monte Carlo with the Experiment 010 winners, 32 s, 10% simulation set |

> [!CAPTION] Table 1. Setup. The simulation set is drawn once before generation 0 and reused by every search.

| Selector | Which images it keeps (per class) |
| --- | --- |
| Stratified random | A random equal share (the control) |
| EL2N | The hardest images for a linear probe on pixels |
| CCS | Error-scored like EL2N, without the hardest 30%, spread over difficulty bands |
| Moderate | Images at the median distance from the class's mean image |
| Moderate coverage | A central distance band, spread out with k-center |

> [!CAPTION] Table 2. The five rules. EL2N, CCS and moderate score raw pixels, not the growing network.

Every rule picks the same number of images per class, so the classes stay balanced; only the choice of images within a class differs. The set is chosen once, before training, and every search in the run reuses it.

## Results

### Final results

**Best overall: CCS**, with the highest mean over all finished runs (75.2% validation, 75.3% test). **For CIFAR-10 alone, Moderate is best** (58.4%), and it is also the most stable there. Moderate coverage is clearly worst and was removed.

| Rank | Selector | CIFAR-10 mean | MNIST mean | **Mean of all runs** |
| ---: | --- | ---: | ---: | ---: |
| 1 | **CCS** | 53.79% | **96.60%** | **75.19%** |
| 2 | Stratified random (control) | 49.83% | 89.13% | 69.48% |
| 3 | EL2N | 55.61% | 77.99% | 66.80% |
| 4 | Moderate | **58.35%** | 74.38% (1 seed) | 63.69% |
| 5 | Moderate coverage | 42.74% | — | 42.74% |

> [!CAPTION] Table 3. Mean final validation accuracy per selector. The last column averages every finished run of the selector (up to 4: two datasets × two seeds), so it mixes the CIFAR-10 and MNIST scales. Test accuracy gives the same order.

![Final validation accuracy of every run per selector](/assets/experiments/011-final-accuracy.svg)

> [!CAPTION] Figure 1. Final validation accuracy (%) of every finished run. The gray line spans the lowest to the highest run, the black point is the mean; circles are CIFAR-10, diamonds MNIST; blue seed 1, orange seed 2.

### CIFAR-10

**Best: Moderate, 58.4%,** with both seeds within 0.9 pp. CCS and stratified random each had one good seed (about 61%) and one poor one (46% and 39%).

| Selector | Mean validation | Seeds | Mean test |
| --- | ---: | ---: | ---: |
| **Moderate** | **58.35%** | 57.9–58.8 | **59.07%** |
| EL2N | 55.61% | 54.6–56.7 | 55.68% |
| CCS | 53.79% | 46.1–61.4 | 53.85% |
| Stratified random | 49.83% | 38.6–61.1 | 49.57% |
| Moderate coverage | 42.74% | 38.7–46.8 | 42.87% |

![CIFAR-10 final accuracy per selector](/assets/experiments/011-final-accuracy-cifar10.svg)

> [!CAPTION] Figure 2. CIFAR-10 final accuracy (%). Bars are two-seed means; circles are validation seeds, diamonds test seeds; blue seed 1, orange seed 2.

### MNIST

**Best: CCS, 96.6%,** with both seeds within 0.3 pp and 7.5 pp ahead of stratified random.

| Selector | Mean validation | Seeds | Mean test |
| --- | ---: | ---: | ---: |
| **CCS** | **96.60%** | 96.4–96.8 | **96.66%** |
| Stratified random | 89.13% | 84.9–93.4 | 89.76% |
| EL2N | 77.99% | 74.2–81.8 | 78.20% |
| Moderate | 74.38% | seed 1 only | 74.58% |

![MNIST final accuracy per selector](/assets/experiments/011-final-accuracy-mnist.svg)

> [!CAPTION] Figure 3. MNIST final accuracy (%), same encoding as Figure 2. Moderate has seed 1 only.

### Training histories

The histories show when the runs separate. A poor run is already behind after the first one to three generations (stratified random seed 1 from generation 1, CCS seed 2 after generation 3) rather than falling behind late, so the early search decisions decide most of the final result.

![Training and validation histories per selector and dataset](/assets/experiments/011-training-histories.svg)

> [!CAPTION] Figure 4. Training (solid) and validation (dashed) accuracy per selector, one row per dataset; blue seed 1, orange seed 2. Dotted lines are generation boundaries (20 epochs). Low runs are low from early generations, not because of a late collapse.

## Conclusion

There is no single best rule for both datasets. CCS has the best average and is the best choice on MNIST; Moderate is the best and most reliable choice on CIFAR-10, the dataset the later experiments use. Both are kept as options, and Experiment 018 (Phase 10) tests them inside the newer search, where they again end up practically equal. The bigger lesson is the seed spread: with two seeds, a difference of a few points between rules is not reliable.

## Limitations

- Two seeds, with ranges up to 22 pp on CIFAR-10; MNIST moderate has one seed; CIFAR-100 was not run.
- Selectors score pixels with a linear probe, not with the network being grown.

## Next

Score the same candidates on the selected subset and on the full validation split to see whether the subset keeps their order.

## Saved results

Script: `experiments/train_cifar10_exp011_simulation_sets.py`. Snapshot: `documentation/website/data/experiments/experiment-011-simulation-sets.json`. Charts: `generate_experiment_011_charts.py`.
