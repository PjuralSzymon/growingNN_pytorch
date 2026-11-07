# Experiment 009: twenty generations on MNIST, CIFAR-10, and CIFAR-100

Three datasets, seeds `1`, `2`, and `3`, twenty generations of ten epochs. Every architecture action stays enabled. The learning rate follows a cosine from `0.1` to `0.001` across the 200 epochs. Monte Carlo search gets 8 seconds, one simulated epoch, and 128 images.

| Dataset | Starter | Parameters |
| --- | --- | ---: |
| MNIST | one 8-channel convolution, then a linear layer to 10 classes | `152` |
| CIFAR-10 | two 128-channel convolutions, hidden width 256, 10 classes | `187,008` |
| CIFAR-100 | the same convolutions, hidden width 512, 100 classes | `268,672` |

Script: `experiments/train_cifar10_exp009_best_reference.py`

Run output: `experiments/output/train_cifar10/runs/exp009_small_20`

Charts: `documentation/website/scripts/generate_experiment_009_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-009-small-20.json`

## Training and validation

![Training and validation accuracy](/assets/experiments/009-small20-train-histories.png)

> [!CAPTION] Figure 1. Solid lines are training accuracy. Dashed lines are validation accuracy. Blue is seed 1, orange is seed 2, green is seed 3. Vertical lines are generation boundaries.

CIFAR-10 training and validation stay together. CIFAR-100 seed 1 training pulls away from validation after epoch 80. MNIST seed 2 training falls when dropout is added; validation does not.

## Official test

![Official test by generation](/assets/experiments/009-small20-test-by-generation.png)

> [!CAPTION] Figure 2. Official test accuracy after each generation. The dashed line on CIFAR-10 is the Experiment 008 sequential-convolution mean, `91.21%`.

| Dataset | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| MNIST | `96.77%` | `98.09%` | `93.99%` | `96.28%` |
| CIFAR-10 | `86.25%` | `86.04%` | `86.82%` | `86.37%` |
| CIFAR-100 | `68.26%` | `66.02%` | `54.00%` | `62.76%` |

CIFAR-10 finishes `4.84` percentage points under the `91.21%` reference, and the three seeds span `0.78` points. CIFAR-100 seed 3 stays about 14 points under the other two.

![Final official test](/assets/experiments/009-small20-final-official-test.png)

> [!CAPTION] Figure 3. Final official test accuracy. One bar is one seed.

## Parameters

![Parameter count](/assets/experiments/009-small20-parameters.png)

> [!CAPTION] Figure 4. Parameter count at the end of each generation, in thousands.

| Dataset | Seed | Final parameters | Reference | Reference size | Times smaller |
| --- | ---: | ---: | --- | --- | ---: |
| MNIST | 1 | `6,088` | LeNet-5, about `99.2%` | 62 thousand | `10.2` |
| MNIST | 2 | `9,016` | LeNet-5, about `99.2%` | 62 thousand | `6.9` |
| MNIST | 3 | `5,793` | LeNet-5, about `99.2%` | 62 thousand | `10.7` |
| CIFAR-10 | 1 | `2,175,744` | ResNet-18, about `95%` | 11.2 million | `5.1` |
| CIFAR-10 | 2 | `1,532,336` | ResNet-18, about `95%` | 11.2 million | `7.3` |
| CIFAR-10 | 3 | `1,867,622` | ResNet-18, about `95%` | 11.2 million | `6.0` |
| CIFAR-100 | 1 | `3,863,705` | ResNet-18, about `76%` | 11.2 million | `2.9` |
| CIFAR-100 | 2 | `3,851,932` | ResNet-18, about `76%` | 11.2 million | `2.9` |
| CIFAR-100 | 3 | `1,834,556` | ResNet-18, about `76%` | 11.2 million | `6.1` |

The reference sizes and accuracies are published counts. They were not trained here. Every grown network is smaller than that reference. CIFAR-10 is about 5 to 7 times smaller and about 9 points below a usual ResNet-18. CIFAR-100 seeds 1 and 2 are about 3 times smaller and about 8 to 10 points below it. Seed 3 is 6 times smaller and about 22 points below it.

## Final graphs

The grown module was not saved. Each drawing is a replay of the saved action list. Dropout is drawn at `p=0.2`. A new convolution is drawn without batch norm when both versions can replay the later actions.

![MNIST final graphs](/assets/experiments/009-small20-arch-mnist.png)

> [!CAPTION] Figure 5. Reconstructed MNIST graphs. `conv1` is the 8-channel starter.

![CIFAR-10 final graphs](/assets/experiments/009-small20-arch-cifar10.png)

> [!CAPTION] Figure 6. Reconstructed CIFAR-10 graphs. `conv1` and `conv2` are the 128-channel starter.

![CIFAR-100 final graphs](/assets/experiments/009-small20-arch-cifar100.png)

> [!CAPTION] Figure 7. Reconstructed CIFAR-100 graphs. The starter convolutions stay 128 channels. The classifier hidden width starts at 512.

All three MNIST seeds keep `conv1`, add more 8-channel convolutions, and finish with a fan of residual linears into the 10-way head. All three CIFAR-10 seeds keep the two-convolution stem and grow mostly residual linears into the classifier. CIFAR-100 seeds 1 and 2 extend that stem with more convolutions. Seed 3 puts a wide residual-linear fan after the pool, and that is the seed that stays at `54.00%` and `1,834,556` parameters.

## Actions that were chosen

![Actions executed by type](/assets/experiments/009-small20-action-counts.png)

> [!CAPTION] Figure 8. Executed actions, summed across seeds 1, 2, and 3. Nineteen actions per seed.

Residual linear is the common choice on CIFAR-10: 26 of 57 actions. MNIST uses residual convolution 17 times and residual linear 15 times. CIFAR-100 splits across residual convolution (13), residual linear (12), and sequential convolution (11). Sequential convolution is used 9 times on MNIST, 11 times on CIFAR-100, and once on CIFAR-10.

## Accuracy change by action type

The change is measured from the end of the generation that committed the action to the end of the next generation. Bars are means. Dots are individual actions. This is the same window as Experiment 001, Figure 8.

![Accuracy change by action type](/assets/experiments/009-small20-action-types.png)

> [!CAPTION] Figure 9. Pooled across MNIST, CIFAR-10, and CIFAR-100. Blue is the training-accuracy change. Green is the validation-accuracy change.

| Action | Observed | Mean training change | Mean validation change |
| --- | ---: | ---: | ---: |
| Sequential conv | `21` | `+7.86 percentage points` | `+7.27 percentage points` |
| Dropout | `12` | `-2.03 percentage points` | `+3.18 percentage points` |
| Add neurons | `17` | `+2.71 percentage points` | `+1.80 percentage points` |
| Delete layer | `5` | `+1.72 percentage points` | `+1.20 percentage points` |
| Sequential linear | `24` | `+2.04 percentage points` | `+1.11 percentage points` |
| Residual conv | `36` | `+1.18 percentage points` | `+0.93 percentage points` |
| Residual linear | `53` | `+1.32 percentage points` | `+0.65 percentage points` |
| Delete neurons | `3` | `+1.58 percentage points` | `-0.00 percentage points` |

Sequential convolution has the highest pooled mean. Nine of those 21 actions are MNIST, and the three points past `+25` percentage points are early MNIST generations, while the 152-parameter starter is still climbing. Dropout lowers training accuracy and raises validation accuracy. Delete neurons is the only type whose mean validation change is not positive, on three observations.

![Accuracy change by action type and dataset](/assets/experiments/009-small20-action-types-by-dataset.png)

> [!CAPTION] Figure 10. The same next-generation change, one panel per dataset. Empty rows are action types that dataset did not execute.

On MNIST, sequential convolution is the strong type (`n=9`, mean validation `+12.83` percentage points) and dropout raises validation while training falls (`n=5`, training `-5.51`, validation `+4.02`). On CIFAR-10 the sequential-convolution mean is one observation, `-2.42` percentage points. Add neurons (`n=9`, `+2.29`) and dropout (`n=3`, `+3.15`) lead there. On CIFAR-100 sequential convolution leads (`n=11`, `+3.59`).

## Largest and smallest validation changes

| Dataset | | Seed | Epochs | Action | Validation change |
| --- | --- | ---: | --- | --- | ---: |
| MNIST | largest | 2 | 10 to 20 | Sequential conv `conv1` → `linear2` | `+30.90 percentage points` |
| MNIST | smallest | 3 | 90 to 100 | Sequential linear `seq_linear_0` → `seq_linear_1` | `-15.60 percentage points` |
| CIFAR-10 | largest | 3 | 40 to 50 | Add neurons on `linear`, ratio `1.1` | `+10.32 percentage points` |
| CIFAR-10 | smallest | 1 | 40 to 50 | Residual conv `conv1` → `seq_linear_0` | `-10.86 percentage points` |
| CIFAR-100 | largest | 1 | 20 to 30 | Sequential conv `conv2` → `linear` | `+9.64 percentage points` |
| CIFAR-100 | smallest | 3 | 50 to 60 | Residual linear `seq_linear_1` → `seq_linear_0` | `-2.50 percentage points` |

Across all three datasets, the largest single change is the MNIST seed 2 sequential convolution, and the smallest is the MNIST seed 3 sequential linear. Both sit on the early or mid climb of a very small network, so the action type and the generation are mixed together in that number.

## What the three seeds show

CIFAR-10 is stable: `86.37%` mean test, graphs of 1.5 to 2.2 million parameters, about 5 to 7 times smaller than ResNet-18, and `4.84` points under the sequential-convolution reference. MNIST stays above `93%` at roughly 6 to 11 times smaller than LeNet-5. CIFAR-100 splits: two seeds near `66%` to `68%` and 3.9 million parameters, one seed at `54.00%` and 1.8 million.

A useful next run is CIFAR-10 with only sequential convolution, the same starter, and the same 200-epoch cosine, compared with `91.21%`.
