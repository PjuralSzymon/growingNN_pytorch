# Experiment 007: CIFAR-10 original growth versus the added 90% settings

This experiment asks whether the original CIFAR-10 growth recipe reaches 90% official test accuracy, and whether the added settings do.

On seeds `110`, `111`, and `112`, the original recipe does not. Its last official test values are `72.79%`, `63.37%`, and `64.20%`. The added settings get much closer. Their best finish-chunk official test values are `90.19%`, `89.65%`, and `89.84%`. Only seed `110` crosses `90%`, and that checkpoint is not the last chunk.

Script: `experiments/train_cifar10_exp007_test90_package.py`

Run output: `experiments/output/train_cifar10/runs/exp007_test90_package`

Charts: `documentation/website/scripts/generate_experiment_007_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-007-cifar10-test90-package.json`

The raw run files are gitignored. Commit the page, the charts, and the snapshot before deleting `experiments/output/`.

## Fixed data setup

Both groups use `Cifar10Data` from `experiments/train_cifar10.py`.

- Official CIFAR-10 test set: `10,000` images. `Cifar10Data.test_loader` is logged only. Search does not read it.
- Train / validation split of the CIFAR-10 training set: `45,000` / `5,000`, `random_split` seed `0`.
- Train augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`.
- Normalization mean `(0.4914, 0.4822, 0.4465)`, std `(0.2023, 0.1994, 0.2010)`.
- Batch size `128`. SGD. Momentum `0.9`.
- Seeds `110`, `111`, and `112` for both groups.

The loop is `train_generations` in `growingnn/training/trainer.py`. One generation trains with `gradient_descent`. Then Monte Carlo search may execute one action. The last generation does not search. Official test accuracy is measured after that generation's SGD and before the action.

## The two groups

| Setting | Original | Added |
| --- | --- | --- |
| Starter | `MinimalCifarNet` | `FourLayerCifarNet` |
| Channels / hidden | `32` / `64` | `64` / `128` |
| Start parameters | `12,832` | `48,448` |
| BatchNorm in the starter | no | `bn1`, `bn2` |
| Generations × epochs | `8` × `10` | `7` × `8`, then a finish stage |
| Learning rate | cosine, peak `0.01` | `0.1` during growth |
| Search gate | slope `3°` | `AlwaysSimulationScheduler` |
| Simulation time / epochs / set | `90 s` / `8` / `2,000` | `8 s` / `1` / `128` |
| Weight decay | `0.0005` | `0.0001` |
| `seq_conv_before_global_pool` | `False` | `True` |
| `seq_conv_bn_relu` | `False` | `True` |
| `max_grad_norm` | unset | `1.0` |
| Structure actions | plain growth: conv and linear adds stay on | `AddSeqConvLayer` only |
| Finish stage | none | `24` chunks × `5` epochs, label smoothing `0.1` |

`running_config_for_group` builds these flags through `experiments_common._running_config`. The product grid in `experiments/train_cifar10.py` is unchanged.

The comparison is the whole original recipe against the whole added group. It is not a one-flag factorial.

## What the added group changes

`GraphStructureQuery.conv_insert_after_node_before_flatten` skips a trailing global pool. `ModelStructureEditor.add_new_seq_layer_before_flatten` then inserts the new conv on the map before that pool when `before_global_pool` is `True`. The original group sets `RunningConfig.seq_conv_before_global_pool` to `False`, so `AddSeqConvLayer._execute` inserts after the pool.

`RunningConfig.seq_conv_bn_relu` stays `False` for the original group. The added group sets it to `True`. `ModelStructureEditor._append_bn_relu` then places `BatchNorm2d` and `F.relu` after each new sequential conv.

`RunningConfig.max_grad_norm` is `1.0` only in the added group. `gradient_descent` clips with that value and skips a non-finite loss.

`finish_added_training` keeps the grown graph and starts a new SGD. The loss is `CrossEntropyLoss(label_smoothing=0.1)`. Accuracy is still argmax. `StepDecayScheduler(at_01=30, at_001=60)` uses finish epochs: `0.1`, then `0.01` from finish epoch `30`, then `0.001` from finish epoch `60`. The function always runs `24` chunks. A chunk at or above `0.90` is not a stop. `added_finish_best.pt` stores the highest chunk-end official test. `added_finish.pt` stores the last chunk.

## Script and timeline

The script runs the original group first, then the added group, one seed at a time. Dates below are filesystem modification times of the saved files, not a training log.

- Original `history.pt`: seed `110` at `2026-10-13 09:35:19`, seed `111` at `10:13:00`, seed `112` at `10:50:56`.
- Added growth checkpoint, seed `110`: `11:17:55`. Its finish file was created at `11:20:39` and last written at `12:23:55`.
- Added seed `112` finish file last written at `2026-10-13 15:31:39`. That is the last file in the run.

The finish stage for seed `110` spans about `63` minutes of wall time for `120` finish epochs. The start of original seed `110` was not written to disk, so the total wall time of the six runs is not known.

## Did either group reach 90% official test?

Official test accuracy is the fraction correct on the `10,000` image test set. For the original group the plotted value is the last generation. For the added group the orange bar is the highest of the `24` finish-chunk measurements, and the pale bar is the last chunk.

![Official test accuracy by seed](/assets/experiments/007-final-official-test.png)

> [!CAPTION] Figure 1. Official test accuracy in percent. Gray is the original group at generation 7. Orange is the best added finish chunk. Pale orange is the last added finish chunk. The dashed line is 90%.

| Seed | Original last test | Original last train | Original last validation | Added end of growth | Added best finish chunk | Added last finish chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `110` | `72.79%` | `75.16%` | `73.48%` | `82.16%` | `90.19%` (chunk `16`, epoch `141`) | `89.98%` |
| `111` | `63.37%` | `63.69%` | `62.72%` | `81.44%` | `89.65%` (chunk `6`, epoch `91`) | `89.41%` |
| `112` | `64.20%` | `64.72%` | `63.36%` | `82.38%` | `89.84%` (chunk `8`, epoch `101`) | `89.66%` |
| Mean | `66.79%` | `67.86%` | `66.52%` | `81.99%` | `89.89%` | `89.68%` |

The original mean stays `23.10` percentage points below the added best-chunk mean. Seed `110` is the only run whose best chunk is at or above `90%`. Its last chunk is `0.21` percentage points lower, at `89.98%`. Seeds `111` and `112` never stored a chunk-end official test at `90%`. Extra finish time after the best chunk did not raise the saved official test.

## How do train, test, and validation move?

Each line is one seed, read from that seed's saved file. Blue is seed `110`, orange is seed `111`, and green is seed `112`.

The left column is the original group, `8` generations × `10` epochs. Train and validation are one point per epoch, `80` points, from `history.pt`. Official test is one point per generation, at epochs `10, 20, …, 80`.

The right column is added seed `110` from `added/seed_110/history.pt`. That file has `176` train points, `176` validation points, and `31` official-test points. The first `56` epochs are growth. The rest are the finish stage. Seeds `111` and `112` do not have this file yet, so they are not drawn on the right.

![Train, test, and validation by group](/assets/experiments/007-history-grid.png)

> [!CAPTION] Figure 2. Three rows: training accuracy, official test accuracy, and validation accuracy, in percent. Left column is the original group, three seeds. Right column is added seed 110 from history.pt. Official test markers are the saved measurements. The dashed line is 90%.

On the left, seed `110` pulls above the other two original seeds after about epoch `30` and stays above them through epoch `80`. On the right, training stays above validation for all `176` epochs. At epoch `56` train is `87.21%` and validation is `81.38%`. At epoch `176` they are `97.94%` and `89.90%`. Official test is `82.23%` at epoch `56`, rises to `89.29%` at epoch `86` while the learning rate is still `0.1`, and then stays near `89.5%` to `89.9%`. The highest saved test is `89.93%` at epoch `161`. The last point is `89.82%`. This curve does not cross `90%`.

Added growth official test by generation:

| Seed | Gen 0 | Gen 1 | Gen 2 | Gen 3 | Gen 4 | Gen 5 | Gen 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `110` | `48.65%` | `69.49%` | `75.54%` | `79.40%` | `81.04%` | `79.40%` | `82.16%` |
| `111` | `60.92%` | `70.24%` | `74.56%` | `80.96%` | `81.32%` | `81.34%` | `81.44%` |
| `112` | `54.51%` | `74.82%` | `80.14%` | `78.54%` | `81.13%` | `83.54%` | `82.38%` |

Generation `6` has no action, because it is the last generation. The finish stage is what moves these curves from about `82%` toward `90%`.

## What did the original search insert?

`history.pt` stores one row per executed action. Validation accuracy is measured on the `5,000` image slice immediately before and after `action.execute`. The next generation's official test is the later recovery window, not this immediate change.

| Seed | Generation | Action | Validation before | Validation after |
| --- | ---: | --- | ---: | ---: |
| `110` | `1` | `AddResConvLayer` | `56.18%` | `55.02%` |
| `110` | `2` | `AddSeqLinearLayer` | `62.48%` | `63.10%` |
| `110` | `3` | `AddResLinearLayer` | `65.10%` | `65.26%` |
| `110` | `4` | `AddResLinearLayer` | `70.20%` | `70.26%` |
| `110` | `5` | `AddResLinearLayer` | `71.34%` | `71.24%` |
| `110` | `6` | `AddSeqLinearLayer` | `73.34%` | `73.18%` |
| `111` | `1` | `AddSeqLinearLayer` | `54.26%` | `55.52%` |
| `111` | `2` | `AddSeqLinearLayer` | `57.32%` | `57.46%` |
| `111` | `3` | `AddResLinearLayer` | `58.30%` | `58.44%` |
| `111` | `4` | `AddResLinearLayer` | `60.34%` | `56.58%` |
| `111` | `5` | `AddSeqLinearLayer` | `62.56%` | `62.62%` |
| `111` | `6` | `AddResLinearLayer` | `62.06%` | `62.18%` |
| `112` | `1` | `AddSeqLinearLayer` | `53.24%` | `54.14%` |
| `112` | `2` | `AddSeqLinearLayer` | `56.86%` | `56.86%` |
| `112` | `3` | `AddResLinearLayer` | `58.00%` | `58.10%` |
| `112` | `4` | `AddResLinearLayer` | `61.02%` | `61.08%` |
| `112` | `5` | `AddSeqLinearLayer` | `61.32%` | `61.66%` |
| `112` | `6` | `AddResLinearLayer` | `61.22%` | `58.88%` |

No generation `0` or generation `7` action is stored. Generation `7` is last, so search does not run. Generation `0` has no stored action either. The slope angle was not saved in `history.pt`.

`AddSeqConvLayer` was legal and was not chosen. The executed actions are residual conv, sequential linear, and residual linear. Three inserts drop validation immediately: seed `110` generation `1` by `1.16` percentage points, seed `111` generation `4` by `3.76` percentage points, and seed `112` generation `6` by `2.34` percentage points. Parameters end at `42,880`, `37,792`, and `37,792`, from `12,832`.

On the original group, train and validation stay close for all `80` epochs. Seed `110` pulls above the other two after about epoch `30` and stays there.

## What did the added group insert?

The checkpoints do not store the action list. `growingnn.log` does. Each row below is one `Generation N action` line from the added runs on `2026-10-13`, matched to the seed by the official-test series in `added_growth.pt`. Validation is the value logged immediately before and after `action.execute`.

| Seed | Generation | Action | From | To | Validation before | Validation after |
| --- | ---: | --- | --- | --- | ---: | ---: |
| `110` | `0` | `AddSeqConvLayer` | `conv2` | `linear` | `48.78%` | `48.78%` |
| `110` | `1` | `AddSeqConvLayer` | `seq_conv_0` | `linear` | `67.82%` | `67.82%` |
| `110` | `2` | `AddSeqConvLayer` | `seq_conv_1` | `linear` | `75.50%` | `75.50%` |
| `110` | `3` | `AddSeqConvLayer` | `seq_conv_1` | `seq_conv_2` | `79.22%` | `79.22%` |
| `110` | `4` | `AddSeqConvLayer` | `seq_conv_0` | `seq_conv_1` | `80.70%` | `80.70%` |
| `110` | `5` | `AddSeqConvLayer` | `seq_conv_0` | `seq_conv_4` | `79.24%` | `79.24%` |
| `111` | `0` | `AddSeqConvLayer` | `conv2` | `linear` | `60.74%` | `60.72%` |
| `111` | `1` | `AddSeqConvLayer` | `seq_conv_0` | `linear` | `70.26%` | `70.26%` |
| `111` | `2` | `AddSeqConvLayer` | `seq_conv_0` | `seq_conv_1` | `74.20%` | `74.16%` |
| `111` | `3` | `AddSeqConvLayer` | `seq_conv_1` | `linear` | `81.36%` | `81.36%` |
| `111` | `4` | `AddSeqConvLayer` | `seq_conv_3` | `linear` | `81.38%` | `81.38%` |
| `111` | `5` | `AddSeqConvLayer` | `seq_conv_0` | `seq_conv_2` | `81.14%` | `81.16%` |
| `112` | `0` | `AddSeqConvLayer` | `conv2` | `linear` | `52.66%` | `52.68%` |
| `112` | `1` | `AddSeqConvLayer` | `seq_conv_0` | `linear` | `75.22%` | `75.20%` |
| `112` | `2` | `AddSeqConvLayer` | `seq_conv_1` | `linear` | `80.32%` | `80.34%` |
| `112` | `3` | `AddSeqConvLayer` | `seq_conv_0` | `seq_conv_1` | `78.82%` | `78.80%` |
| `112` | `4` | `AddSeqConvLayer` | `seq_conv_2` | `linear` | `81.02%` | `81.02%` |
| `112` | `5` | `AddSeqConvLayer` | `seq_conv_1` | `seq_conv_2` | `83.84%` | `83.84%` |

Every seed ends with `seq_conv_0` through `seq_conv_5`, each with a matching BatchNorm. That is six `AddSeqConvLayer` inserts. Each conv weight is shaped `(64, 64, 3, 3)`, so the insert is a `3×3` map at the pre-pool resolution. End parameter count is `271,816` on every seed, from a `48,448` parameter `FourLayerCifarNet`. The finish stage does not add modules. The best and last checkpoints have the same module names. Immediate validation barely moves. The later official-test gain is the next generation of training, not the insert itself.

## Limitations

- Three seeds. The original last-test spread is `9.42` percentage points (`72.79%` to `63.37%`). The added best-chunk spread is `0.54` percentage points (`90.19%` to `89.65%`).
- Figure 1 still uses the first three-seed run. Its finish series between chunks was not saved, so the bars are the best and last checkpoints only.
- Added seed `110` now has per-epoch train and validation in `history.pt`. Figure 2 uses that file. Seeds `111` and `112` do not have it yet. The right-hand curve is this later seed `110` run. Its best official test is `89.93%`, so it is not the `90.19%` checkpoint in Figure 1.
- The two groups differ in starter, learning rate, search budget, action set, and the finish stage together. A single flag is not identified.
- Official test accuracy was not an input to search.

## Conclusions

1. The original recipe does not reach `90%` official test accuracy. The three-seed mean at the last generation is `66.79%`.
2. The added package reaches a three-seed best-chunk mean of `89.89%`. Only seed `110` crosses `90%`, at `90.19%` on finish chunk `16`.
3. Running the finish stage to the end does not hold that peak. The last-chunk mean is `89.68%`. Seed `110` ends at `89.98%`.
4. The `90%` region is reached in the finish stage, not in the seven growth generations. Those end at a mean official test of `81.99%`.

## Next experiments

Save the finish-chunk official test list, and the added action endpoints, in the run folder. The next report can then show the finish curve instead of two checkpoints.

Repeat the added group with the finish stage stopped at the best chunk, and with one change at a time: before-pool insert, BatchNorm-ReLU, the gradient clip, and the finish schedule. This page only measures them together.
