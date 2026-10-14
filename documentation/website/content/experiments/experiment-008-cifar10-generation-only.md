# Experiment 008: CIFAR-10 generation loop without a finish stage

This experiment asks what official test accuracy `train_generations` reaches when training stays inside the generation loop.

On seeds `110`, `111`, and `112`, the last official test values are `91.11%`, `91.17%`, and `91.36%`. The mean is `91.21%`. For every seed the highest of the `12` official-test measurements is that last value. None of the saved official-test points reaches `92%`.

Script: `experiments/train_cifar10_exp008_generation_only.py`

Run output: `experiments/output/train_cifar10/runs/exp008_generation_only`

Charts: `documentation/website/scripts/generate_experiment_008_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-008-cifar10-generation-only.json`

The raw run files are gitignored. Commit the page, the charts, and the snapshot before deleting `experiments/output/`.

## Fixed data setup

The run uses `Cifar10Data` from `experiments/train_cifar10.py`.

- Official CIFAR-10 test set: `10,000` images. `Cifar10Data.test_loader` is logged only. Search does not read it.
- Train / validation split of the CIFAR-10 training set: `45,000` / `5,000`, `random_split` seed `0`.
- Train augmentation: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`.
- `with_cutout` adds `RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)` on the training loader only.
- Normalization mean `(0.4914, 0.4822, 0.4465)`, std `(0.2023, 0.1994, 0.2010)`.
- Batch size `128`. SGD. Momentum `0.9`. Weight decay `0.0001`.
- Seeds `110`, `111`, and `112`.

The loop is `train_generations` in `growingnn/training/trainer.py`. One generation trains with `gradient_descent` for `10` epochs. Monte Carlo may then execute one action. Generation `11` does not search. Official test accuracy is measured after that generation's SGD and before the action.

`history.pt` has `120` training points, `120` validation points, and `12` official-test points. `grown.pt` stores the final `state_dict` and the same official-test list.

## Settings

`run_seed` starts from `added_hyperparameters` and `running_config_for_group("added")` in `experiments/train_cifar10_exp007_test90_package.py`, then replaces the generation budget, the width, the learning-rate schedule, and the loss.

| Setting | Value |
| --- | --- |
| Starter | `FourLayerCifarNet` |
| Channels / hidden | `128` / `256` |
| Start parameters | `187,008` |
| Generations × epochs | `12` × `10` |
| Learning rate | `CosineDecayScheduler(120)`, from `0.1` down to `0.001` |
| Loss | `CrossEntropyLoss(label_smoothing=0.1)` for all `120` epochs |
| Accuracy | argmax, on train, validation, and official test |
| Stopper | `AccuracyStopper(0.999)` |
| Search gate | `AlwaysSimulationScheduler` |
| Simulation time / epochs / set | `8 s` / `1` / `128` |
| Score | `SimulationScore(weight_acc=1, weight_countW=0)` |
| `seq_conv_before_global_pool` | `True` |
| `seq_conv_bn_relu` | `True` |
| `max_grad_norm` | `1.0` |
| Enabled structure action | `AddSeqConvLayer` only |
| Finish stage | `finish_added_training` is not called |

`CosineDecayScheduler.structure_changed` does not reset the epoch counter. `freeze_global_schedule_progress` keeps simulation scoring from advancing that counter. The saved learning rate is the same list on all three seeds. It is `0.1` at epoch `1`, `0.0512` at epoch `60`, `0.0157` at epoch `90`, and `0.001` at epoch `120`.

The flags in `_ADDED_OFF_FLAGS` turn off delete-layer, delete-neuron, add-neuron, dropout, sequential linear, residual linear, and residual conv. `AddSeqConvLayer` stays on.

## How this differs from Experiment 007

Experiment 007 compares an original recipe with an added group. The added group grows for `7` × `8` epochs and then calls `finish_added_training` for `24` chunks of `5` epochs. Its published best finish-chunk official tests are `90.19%`, `89.65%`, and `89.84%`. The mean is `89.89%`. Only seed `110` is at or above `90%`.

Experiment 008 is a separate script. It does not rerun the original group, and it does not edit Experiment 007. Relative to that added group, several settings change together:

- The budget is `12` × `10` epochs inside `train_generations`, and the run ends when that call returns.
- The starter is wider: `128` channels and hidden size `256`, instead of `64` and `128`.
- The learning rate is one cosine over the `120` epochs, instead of a step schedule held at `0.1` during growth and `StepDecayScheduler(30, 60)` in the finish stage.
- Cutout is on the training loader.
- Label smoothing `0.1` is on for the whole run. In Experiment 007 it starts in `finish_added_training`.

The last-official-test mean here is `1.32` percentage points above the Experiment 007 best-chunk mean. That gap is the whole recipe, not one flag.

## Script and timeline

`main` runs seeds `110`, `111`, and `112` in that order. Dates below are filesystem modification times of `history.pt`, not a training log. The device is not stored in the file. `run_seed` uses CUDA when `torch.cuda.is_available()` is true, and CPU otherwise.

- Seed `110`: `2026-10-14 08:40:11`.
- Seed `111`: `2026-10-14 09:12:50`. That is `32` minutes `39` seconds after seed `110`.
- Seed `112`: `2026-10-14 09:44:15`. That is `31` minutes `25` seconds after seed `111`.

The start of seed `110` was not written to disk, so the total wall time is not known.

## What official test did the generation loop reach?

Official test accuracy is the fraction correct on the `10,000` image test set. Each marker is one generation end, at epochs `10, 20, …, 120`.

![Official test accuracy by generation](/assets/experiments/008-official-test.png)

> [!CAPTION] Figure 1. Official test accuracy in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Each marker is the official test at the end of one generation. The dashed line is 90%. The vertical axis starts at 50%.

| Generation | Epoch | Seed `110` | Seed `111` | Seed `112` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `10` | `59.27%` | `58.32%` | `63.35%` |
| `1` | `20` | `76.77%` | `75.48%` | `72.70%` |
| `2` | `30` | `77.73%` | `81.06%` | `80.58%` |
| `3` | `40` | `81.65%` | `83.80%` | `83.74%` |
| `4` | `50` | `85.43%` | `86.51%` | `84.54%` |
| `5` | `60` | `86.42%` | `86.37%` | `87.27%` |
| `6` | `70` | `88.39%` | `88.20%` | `88.09%` |
| `7` | `80` | `89.56%` | `88.80%` | `88.75%` |
| `8` | `90` | `89.83%` | `90.32%` | `89.69%` |
| `9` | `100` | `90.55%` | `90.82%` | `90.64%` |
| `10` | `110` | `90.93%` | `90.93%` | `91.28%` |
| `11` | `120` | `91.11%` | `91.17%` | `91.36%` |

Seed `111` first reaches `90%` at generation `8`. Seeds `110` and `112` first reach it at generation `9`. Seed `111` is also the only decrease: generation `4` is `86.51%` and generation `5` is `86.37%`, a drop of `0.14` percentage points. The other two official-test series rise at every generation.

| Seed | Last train | Last validation | Last official test |
| --- | ---: | ---: | ---: |
| `110` | `96.61%` | `91.60%` | `91.11%` |
| `111` | `96.38%` | `91.28%` | `91.17%` |
| `112` | `96.37%` | `90.76%` | `91.36%` |
| Mean | `96.45%` | `91.21%` | `91.21%` |

The last official-test values sit in a band of `0.25` percentage points. The last validation values sit in a band of `0.84` percentage points. Seed `112` has the highest official test and the lowest validation. Its validation is `0.60` percentage points below its official test. Seed `110` validation is `0.49` percentage points above its official test.

## How do train, test, and validation move?

Each line is one seed, read from that seed's `history.pt`.

![Train, test, and validation](/assets/experiments/008-history.png)

> [!CAPTION] Figure 2. Training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. The vertical axis starts at 30%.

![Learning rate](/assets/experiments/008-learning-rate.png)

> [!CAPTION] Figure 3. Learning rate by epoch. The three history files store the same learning-rate list, so one line is drawn.

At epoch `1`, training accuracy is about `33%` and validation is higher: `39.72%`, `37.36%`, and `42.30%`. Validation stays at or above training for `8`, `15`, and `10` epochs on seeds `110`, `111`, and `112`. Those epochs are mostly in the first part of the run. After that, training stays above validation. At epoch `120` the train-minus-validation gap is `5.01`, `5.10`, and `5.61` percentage points.

Official test at epoch `10` still differs by seed. Seed `112` is `63.35%` and seed `111` is `58.32%`, a gap of `5.03` percentage points. By epoch `120` that gap is `0.25` percentage points.

The markers keep rising after the learning rate has fallen below `0.02`, and the steps get small. From epoch `100` to epoch `120` the official-test gains are `0.56`, `0.35`, and `0.72` percentage points. Over those same epochs the parameter count goes from `1,517,568` to `1,813,248`.

Seed `110` validation peaks at epoch `116` at `91.98%`, then ends at `91.60%`, while its official test is still highest at epoch `120`. Seed `112` validation peaks at epoch `108` at `91.02%` and ends at `90.76%`. Seed `111` validation is highest at epoch `120`.

## What did search insert?

`history.pt` stores one row per executed action. Validation accuracy is measured on the `5,000` image slice immediately before and after `action.execute`. The next generation's official test is the later window, not this immediate change.

Every executed action is `AddSeqConvLayer`. There are `11` of them, at generations `0` through `10`. Generation `11` has no action. The final `state_dict` contains `seq_conv_0` through `seq_conv_10`. The history does not store which layers the new conv was tied to.

The immediate validation change is at most `0.08` percentage points. The largest is seed `110`, generation `0`, from `58.80%` to `58.88%`. The other immediate changes are `0.04` percentage points or smaller. The jump in official test from generation `0` to generation `1` is `17.50`, `17.16`, and `9.35` percentage points. That jump is the next `10` epochs, not the immediate insert.

`param_count` is recorded after the generation's SGD and before the action, then copied onto every epoch of that generation. It stays `187,008` through epoch `10`. Each later generation is `147,840` parameters higher. Epoch `120` is `1,813,248` on every seed. `147,840` is one `128`-channel `3×3` convolution, its bias, and one `BatchNorm2d` of `128` channels.

## Limitations and seed effects

- Three seeds. Last official test agrees within `0.25` percentage points. Last validation does not: seed `112` is the low validation and the high official test.
- Official test is measured `12` times. A rise or drop between those epochs is not in the file.
- The enabled action list contains only `AddSeqConvLayer`. The run does not show a choice among action families. It can still choose among insert positions, and those positions are not in `history.pt`.
- The comparison with Experiment 007 changes width, epoch budget, learning-rate schedule, cutout, label smoothing, and the finish stage together.
- `history.pt` does not record the device. The wall-clock start of seed `110` is not in the file.

## Conclusions

`train_generations` with this recipe reaches a last official test of `91.11%`, `91.17%`, and `91.36%`. The best saved official test is generation `11` for every seed. The same runs leave training accuracy near `96.5%` and validation near `91.2%`.

Another same-width sequential convolution late in the cosine adds `147,840` parameters and a fraction of a percentage point of official test. From epoch `100` to epoch `120` two such convolutions add `295,680` parameters and at most `0.72` percentage points of official test.

## Next experiments

A later run that wants official test above this band needs a different growth step than another `128`-channel sequential convolution at the end of the cosine. Experiment 007 can stay as the finish-stage comparison. This page does not replace it.
