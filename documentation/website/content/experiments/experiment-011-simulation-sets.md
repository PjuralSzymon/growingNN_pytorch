# Experiment 011: simulation-set selectors

## Introduction

Experiment 010 found its largest improvement by enlarging the simulation set—the small subset used to train and score candidate architectures. Until now, we had created this set by randomly sampling data examples from the training dataset, with the same number from each class. We continued from 010's selected search settings and 10% subset size, but now asked whether a different way of choosing those examples would help. We tested four alternatives against the random sampling already in use.

## Setup

| Setting | Value |
| --- | --- |
| Datasets, seeds | CIFAR-10 and MNIST, seeds 1 and 2 (MNIST moderate: seed 1 only) |
| Starter | Conv →8, ReLU, pool, Linear 8→10; **296 parameters on CIFAR-10, 152 on MNIST** |
| Schedule | 10 generations × 20 epochs |
| Search | Monte Carlo with the Experiment 010 winners, 32 s, 10% simulation set |

> [!CAPTION] Table 1. Setup. The simulation set is drawn once before generation 0 and reused by every search.

| Simulation-set generator | Algorithm |
| --- | --- |
| Stratified random (existing method) | Take a random sample from the training dataset, with the same number of examples from each class. |
| EL2N | Train a simple linear classifier on the examples' feature vectors, then select the examples on which it makes the largest prediction errors. |
| CCS | Use the same prediction errors to estimate difficulty. Exclude the hardest 30%, then sample from across the remaining difficulty range. |
| Moderate | Compute the mean feature vector for each class. Select examples whose distance from that mean is closest to the class's median distance. |
| Moderate coverage | Start with examples around that median distance, then select examples far from those already chosen to avoid filling the set with similar data. |

> [!CAPTION] Table 2. Five ways to build the simulation set. All keep classes balanced. EL2N and CCS use a separate linear classifier, not the network being grown.

The generators that compare or score examples use a **data matrix**: one row per example and one column per numerical feature. In these runs, each input was flattened into a vector. Other data types can use a suitable numerical representation; the selection rules operate on those vectors and class labels.

“Stratified” simply means sampling separately within each class to keep those counts balanced. Each generator also selects a separate subset of validation examples for scoring. Both subsets are created before training and reused throughout the run.

## 1. Comparing selectors on CIFAR-10

On CIFAR-10, stratified random sampling is the control. The four alternatives change which examples each class contributes, while keeping the subset size and search settings fixed.

**Best: Moderate, 58.4%,** with both seeds within 0.9 pp. CCS and stratified random each had one good seed (about 61%) and one poor one (46% and 39%).

![CIFAR-10 final accuracy per selector](/assets/experiments/011-final-accuracy-cifar10.svg)

> [!CAPTION] Figure 1. CIFAR-10 final accuracy (%). Bars are two-seed means; circles are validation seeds, diamonds test seeds; blue seed 1, orange seed 2.

| Selector | Mean validation | Seeds | Mean test |
| --- | ---: | ---: | ---: |
| **Moderate** | **58.35%** | 57.9–58.8 | **59.07%** |
| EL2N | 55.61% | 54.6–56.7 | 55.68% |
| CCS | 53.79% | 46.1–61.4 | 53.85% |
| Stratified random | 49.83% | 38.6–61.1 | 49.57% |
| Moderate coverage | 42.74% | 38.7–46.8 | 42.87% |

> [!CAPTION] Table 3. CIFAR-10 final accuracy per selector: mean of two seeds, with the two seeds' validation results. Bold is the best.

## 2. Checking whether the ranking holds on MNIST

The same rules are also tested on MNIST to check whether the CIFAR-10 ranking generalizes.

**Best: CCS, 96.6%,** with both seeds within 0.3 pp and 7.5 pp ahead of stratified random.

![MNIST final accuracy per selector](/assets/experiments/011-final-accuracy-mnist.svg)

> [!CAPTION] Figure 2. MNIST final accuracy (%). Bars are means; circles are validation seeds, diamonds test seeds. Moderate has seed 1 only.

| Selector | Mean validation | Seeds | Mean test |
| --- | ---: | ---: | ---: |
| **CCS** | **96.60%** | 96.4–96.8 | **96.66%** |
| Stratified random | 89.13% | 84.9–93.4 | 89.76% |
| EL2N | 77.99% | 74.2–81.8 | 78.20% |
| Moderate | 74.38% | seed 1 only | 74.58% |

> [!CAPTION] Table 4. MNIST final accuracy per selector: mean of the finished seeds, with the seeds' validation results. Bold is the best.

## 3. Summary

The combined view below averages the final accuracy of every completed run for each generator, giving each run equal weight.

![Mean final validation and test accuracy for all simulation-set generators](/assets/experiments/011-summary-means.svg)

> [!CAPTION] Figure 3. Mean final accuracy across completed CIFAR-10 and MNIST runs. Bars show validation and test separately; labels below each generator show how many runs came from each dataset.

| Simulation-set generator | CIFAR-10 runs | MNIST runs | Mean validation | Mean test |
| --- | ---: | ---: | ---: | ---: |
| **CCS** | 2 | 2 | **75.19%** | **75.26%** |
| Stratified random | 2 | 2 | 69.48% | 69.66% |
| EL2N | 2 | 2 | 66.80% | 66.94% |
| Moderate | 2 | 1 | 63.69% | 64.24% |
| Moderate coverage | 2 | 0 | 42.74% | 42.87% |

> [!CAPTION] Table 5. Means from the saved final results. Moderate has only one MNIST seed; Moderate coverage has no MNIST run, so their combined means are not directly comparable with the complete four-run groups.

**CCS has the highest combined mean in the completed runs.** It also wins on MNIST, while Moderate wins on CIFAR-10. We therefore kept both as options and removed Moderate coverage. The combined average summarizes these runs; it does not establish one best generator for every dataset. Experiment 013 revisits the choice inside the revised search.

### How CCS builds the simulation set

CCS means **coverage-centric selection**. The idea is that a small subset should *cover* the whole range of examples, from easy to fairly hard, instead of keeping only one kind. To do that it first needs a difficulty score for every image.

**What "hard" and "easy" mean here.** A small separate linear classifier (not the network being grown) is trained on the training images for 10 epochs. After every epoch it predicts each image's class, and the size of its mistake is measured:

- An **easy** image is one the classifier gets right with high confidence in every epoch: its error stays close to 0.
- A **hard** image is one it keeps getting wrong or keeps doubting: its error stays high, up to 2 for a confident wrong answer.

So "hard" means "this simple classifier has persistently high error on it", not a property of the image itself. The hardest images are often unusual, ambiguous or mislabelled.

```text
1. Difficulty score of every training image
   Train a linear classifier on the training images for 10 epochs.
   After each epoch, for each image:
       p      = predicted probability of every class
       target = 1 for the correct class, 0 for the others
       error  = sum over classes of (p - target)²
   difficulty = the image's average error over the 10 epochs
       (0 = always right and sure  -> easy;  about 2 = always wrong and sure -> hard)

2. For each class separately (every class gets the same number of images, its quota):
   a. Sort the images of the class from hardest to easiest.
   b. Drop the hardest 30%, the images most likely to be ambiguous or mislabelled
      (fewer if dropping 30% would leave less than the quota).
   c. Split the remaining difficulty range, from the easiest to the hardest kept score,
      into 50 bands of equal width; ignore empty bands.
   d. Fill the quota band by band, starting from the band with the fewest images:
          share = remaining quota / number of bands still left
          take `share` random images from the band (or all of them if it has fewer)
      Places a small band cannot fill pass on to the larger bands.

3. The simulation set is the union of all classes.
```

**Example of the error.** With two classes and the first one correct, a prediction of `[0.9, 0.1]` gives an error of (0.9 − 1)² + (0.1 − 0)² = **0.02**: easy. A prediction of `[0.2, 0.8]` gives (0.2 − 1)² + (0.8 − 0)² = **1.28**: hard.

**Why bands of equal width.** Most images have low difficulty, so a random sample would be mostly easy images. Equal-width bands give each difficulty level roughly the same share of the subset, so easy, medium and fairly hard images are all represented. EL2N, by contrast, keeps only the hardest images, and Stratified random keeps a random mix.

The validation subset is chosen the same way, using the trained classifier's predictions on the validation images (it is not trained on them). Both subsets stay fixed for the whole growing run.
