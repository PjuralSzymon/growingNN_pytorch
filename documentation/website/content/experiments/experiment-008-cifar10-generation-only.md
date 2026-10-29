# Experiment 008: CIFAR-10 generation loop without a finish stage

This page has eleven sections, in this order.

1. [Improvements that make CIFAR-10 work in the growing algorithm](#improvements-that-make-cifar-10-work-in-the-growing-algorithm). How the recipe worked before, and how it works now.
2. [Phase 1: sequential convolution only](#phase-1-sequential-convolution-only). The finished run that allows only `AddSeqConvLayer`. Last official test is `91.11%`, `91.17%`, and `91.36%`. The mean is `91.21%`.
3. [Phase 2: which actions search can choose](#phase-2-which-actions-search-can-choose). The same recipe with three action lists. Last-official-test means are `85.68%` for `all`, `86.21%` for `grow`, and `83.68%` for `seq_res`.
4. [Phase 3: how long search looks](#phase-3-how-long-search-looks). The all-action list with a `100` second simulation. Last official test is `88.45%`, `85.04%`, and `89.35%`. The mean is `87.61%`.
5. [Phase 4: full augmented grade](#phase-4-full-augmented-grade). Search grades every action on all `45,000` augmented training images. Only the `1` epoch grade finished. Last official test is `88.58%`, `88.15%`, and `87.57%`. The mean is `88.10%`. The `2` epoch and `4` epoch grades have no `history.pt`, because that amount of data took too much time.
6. [Phase 5: first 5,000 clean images](#phase-5-first-5000-clean-images). The same search, graded on the first `5,000` clean images, at simulation lengths `1`, `2`, `4`, `5`, `6`, and `7`. All six grades finished. The highest mean is the `4` epoch grade, `87.69%`.
7. [Phase 6: smaller clean prefixes](#phase-6-smaller-clean-prefixes). The `4` epoch grade, with the clean prefix halved six times: `5,000`, `2,500`, `1,250`, `625`, `312`, `156`, and `78`. All seven grades finished. The means are `85.65%`, `89.01%`, `88.24%`, `84.48%`, `81.03%`, `83.62%`, and `83.20%`. The `2,500` image mean is the highest.
8. [Phase 7: longer real training](#phase-7-longer-real-training). The same `1,250` image grade at `10`, `20`, and `30` real epochs per generation. The means are `88.24%`, `84.65%`, and `87.48%`. The `10` epoch mean is the highest.
9. [Phase 8: how many generations](#phase-8-how-many-generations). The same `1,250` image grade at `10` real epochs, with `10`, `12`, `15`, and `20` generations. The means are `80.65%`, `88.24%`, `90.04%`, and `88.35%`. The `15` generation mean is the highest.
10. [Phase 10: learning rate and conv placement](#phase-10-learning-rate-and-conv-placement). The `15` generation grade, with the learning rate and the conv placement turned on and off. Every arm uses all actions. Five seeds. The means are `82.17%`, `83.23%`, `83.33%`, and `88.04%`. The arm with both changes is the highest.
11. [Phase 11: one change at a time](#phase-11-one-change-at-a-time). The phase 10 `base` recipe, with one remaining change turned off per arm. Seeds `110` and `111`. The new cells are empty until the runs finish.

Script: `experiments/train_cifar10_exp008_generation_only.py`

Phase 1 output: `experiments/output/train_cifar10/runs/exp008_generation_only`

Phase 2 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase2`

Phase 3 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase3`

Phase 4 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase4`

Phase 5 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase5`

Phase 6 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase6`

Phase 7 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase7`

Phase 8 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase8`

Phase 10 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase10`

Phase 11 output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase11`

Charts: `documentation/website/scripts/generate_experiment_008_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-008-cifar10-generation-only.json`

The raw run files are gitignored. Commit the page, the charts, and the snapshot before deleting `experiments/output/`.

## Improvements that make CIFAR-10 work in the growing algorithm

Experiment 007's original group is the run that stayed low. Its last official-test mean is `66.79%`. Experiment 008 ends at `91.21%`. The points below are the changes between those two recipes. They were applied together. This page does not separate which single change caused the gain.

### Starter network

How it was working. The original group starts from `MinimalCifarNet`. The forward pass is `conv1` (`3` to `32`), ReLU, max-pool, `conv2` (`32` to `32`), ReLU, max-pool, global pool, `linear` (`32` to `64`), ReLU, `linear2` (`64` to `10`). There is no BatchNorm. The saved start parameter count is `12,832`. Before that plain net, the CIFAR script started from a tiny residual block with BatchNorm, and it used the official test set as the validation set.

How it is working now. Experiment 008 starts from `FourLayerCifarNet` with `128` channels and hidden size `256`. The four learnable layers are `conv1` (`3` to `128`), `conv2` (`128` to `128`), `linear` (`128` to `256`), and `linear2` (`256` to `10`). `bn1` sits on `conv1` and `bn2` sits on `conv2`. Each convolution is followed by ReLU, then a max-pool. A global pool feeds the two linear layers. The saved start parameter count is `187,008`. The BatchNorm layers and the wider maps give the later `3×3` inserts a real feature map to stack on. This run ends at `1,813,248` parameters.

### Where a new convolution is inserted

How it was working. A new sequential convolution was placed on the output of the global pool. That map is `1×1`. The original group in Experiment 007 still uses that placement, because `seq_conv_insert_before_global_pool` is `False`.

How it is working now. `GraphStructureQuery.conv_insert_after_node_before_flatten` skips the trailing global pool. The new convolution runs at `8×8`, after the two max-pools and before the global pool. `seq_conv_insert_before_global_pool` is `True`. The convolution still starts as an eye, so the network function barely changes at the moment of the insert.

### BatchNorm and ReLU after each new convolution

How it was working. The new convolution was only the eye convolution. A second eye convolution, with nothing between the two, stayed a linear copy of the first.

How it is working now. Search is offered both variants. One is the convolution alone. The other uses `_append_batchnorm_and_relu`, which places `BatchNorm2d` and ReLU after the new sequential convolution. Each inserted convolution that takes the second variant has its own BatchNorm. The saved step for that variant is `147,840` parameters, which is one `128`-channel `3×3` convolution, its bias, and one `BatchNorm2d`.

### Which actions search can choose

How it was working. The original group leaves sequential convolution, sequential linear, residual linear, and residual convolution enabled. Search chose the linear and residual actions. `AddSeqConvLayer` was legal and was not chosen. The three networks ended at `42,880`, `37,792`, and `37,792` parameters.

How it is working now. Delete-layer, delete-neuron, add-neuron, dropout, sequential linear, residual linear, and residual conv are off. The only enabled structure action is `AddSeqConvLayer`. Every executed action in this run is that action. There are `11` of them, at generations `0` through `10`. Generation `11` does not search.

### When search runs, and how long it looks

How it was working. The original group uses a slope gate of `3` degrees. Search runs only when the training-accuracy slope is close to flat. Its simulation budget is `90` seconds, `8` epochs, and `2,000` images. Generation `0` stored no action.

How it is working now. `AlwaysSimulationScheduler` runs search after every generation except the last. The budget is `8` seconds, `1` epoch, and `128` images. The score is still accuracy weight `1` and parameter-count weight `0`. The algorithm is still Monte Carlo. Phase 1 and phase 2 use this short budget. Phase 3 keeps this gate and the all-action list, and changes the budget to `100` seconds, `8` epochs, and `2,000` images.

### Learning rate

How it was working. The original group uses a composed cosine. The global curve goes from `0.01` down to `0.0001` over `80` epochs. After an action, `ComposedLearningRateScheduler.structure_changed` restarts a logistic warmup of `10` iterations, and that factor multiplies the cosine. The added group of Experiment 007 instead holds `0.1` for `56` growth epochs, then steps to `0.01` and `0.001` during the finish stage.

How it is working now. `CosineDecayScheduler(120)` goes from `0.1` at epoch `1` to `0.001` at epoch `120`. An action does not restart it. `freeze_global_schedule_progress` keeps simulation scoring from advancing the epoch counter. The same learning-rate list is stored for all three seeds.

### Momentum, weight decay, and the gradient clip

How it was working. `gradient_descent` built SGD with momentum `0.0` and weight decay `0.0`, and it did not clip. The original CIFAR group later sets momentum `0.9` and weight decay `0.0005`, and it still leaves the clip unset. The `RunningConfig` defaults remain momentum `0.0` and weight decay `0.0`, which is what MNIST uses.

How it is working now. Experiment 008 uses momentum `0.9`, weight decay `0.0001`, and `max_grad_norm` `1.0`. `gradient_descent` clips with that value. A loss that is not finite is skipped.

### Validation and the official test

How it was working. The older CIFAR loader used the official `10,000` image test set as validation. Search and the stopper saw the test set. The summary text called that number `val_acc`.

How it is working now. Validation is `5,000` images taken from the `50,000` training images, with `random_split` generator seed `0`. Training uses the other `45,000`. `Cifar10Data.test_loader` loads the official test set only to log it. Search does not read it. Experiment 007 and Experiment 008 both use this split.

### Length of training

How it was working. The original group trains `8` generations of `10` epochs and then stops. The added group trains `7` generations of `8` epochs and then calls `finish_added_training` for `24` chunks of `5` epochs on the frozen graph. That finish stage is `120` extra epochs. Its published best official tests are `90.19%`, `89.65%`, and `89.84%`.

How it is working now. Experiment 008 stays inside `train_generations` for `12` generations of `10` epochs, `120` epochs in total. It does not call `finish_added_training`. The last official tests are `91.11%`, `91.17%`, and `91.36%`.

### Cutout and label smoothing

How it was working. Training augmentation is crop and horizontal flip. The loss is `CrossEntropyLoss()` with no label smoothing. Experiment 007 turns smoothing on only inside the finish stage of the added group.

How it is working now. `with_cutout` adds `RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)` on the training loader only. Validation and the official test are not erased. `CrossEntropyLoss(label_smoothing=0.1)` is the loss for all `120` epochs. The reported accuracy is still the argmax match.

## Phase 1: sequential convolution only

Phase 1 allows only `AddSeqConvLayer`. On seeds `110`, `111`, and `112`, the last official test values are `91.11%`, `91.17%`, and `91.36%`. The mean is `91.21%`. For every seed the highest of the `12` official-test measurements is that last value. None of the saved official-test points reaches `92%`.

### Settings

| Setting | Value |
| --- | --- |
| Data | `Cifar10Data` |
| Official test | `10,000` images, logged only. Search does not read it. |
| Train / validation | `45,000` / `5,000`, `random_split` seed `0` |
| Train augmentation | `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)` |
| Normalization | mean `(0.4914, 0.4822, 0.4465)`, std `(0.2023, 0.1994, 0.2010)` |
| Batch / optimizer | `128`. SGD, momentum `0.9`, weight decay `0.0001` |
| Seeds | `110`, `111`, `112` |
| Starter | `FourLayerCifarNet`, channels `128`, hidden `256`, start `187,008` parameters |
| Generations × epochs | `12` × `10` inside `train_generations`. Generation `11` does not search. |
| Official test timing | After that generation's SGD and before the action |
| Learning rate | `CosineDecayScheduler(120)`, from `0.1` down to `0.001`. An action does not restart it. |
| Loss / accuracy | `CrossEntropyLoss(label_smoothing=0.1)`. Accuracy is argmax. |
| Stopper | `AccuracyStopper(0.999)` |
| Search | `AlwaysSimulationScheduler`, `8` s / `1` epoch / `128` images |
| Score | `SimulationScore(weight_acc=1, weight_countW=0)` |
| New convolution | Before the global pool, with BatchNorm and ReLU |
| Gradient clip | `1.0` |
| Enabled action | `AddSeqConvLayer` only |
| Finish stage | `finish_added_training` is not called |

### How the parameter count grew

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. The new convolution shows up in the next generation.

![Parameter count by epoch](/assets/experiments/008-parameters.png)

> [!CAPTION] Figure 1. Parameter count by epoch. Seeds 110, 111, and 112 stored the same list, so one line is drawn. The line is flat inside a generation and steps up by 147,840 parameters at the first epoch after each action.

| Seed | Epochs `1` to `10` | Epoch `120` | Sequential convolutions added |
| --- | ---: | ---: | ---: |
| `110` | `187,008` | `1,813,248` | `11` |
| `111` | `187,008` | `1,813,248` | `11` |
| `112` | `187,008` | `1,813,248` | `11` |

The count stays `187,008` through epoch `10`. It then rises at epochs `11, 21, 31, 41, 51, 61, 71, 81, 91, 101,` and `111`. Epoch `120` is `1,813,248` on every seed. Generation `11` adds no parameters.

### What official test did the generation loop reach?

Official test accuracy is the fraction correct on the `10,000` image test set. The `12` measurements are the generation ends, at epochs `10, 20, …, 120`. Those markers are the middle column of Figure 2.

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

### How do train, test, and validation move?

Each line is one seed, read from that seed's `history.pt`.

![Train, test, and validation](/assets/experiments/008-history.png)

> [!CAPTION] Figure 2. One row for only convolution. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

![Learning rate](/assets/experiments/008-learning-rate.png)

> [!CAPTION] Figure 3. Learning rate by epoch. The three history files store the same learning-rate list, so one line is drawn.

At epoch `1`, training accuracy is about `33%` and validation is higher: `39.72%`, `37.36%`, and `42.30%`. Validation stays at or above training for `8`, `15`, and `10` epochs on seeds `110`, `111`, and `112`. Those epochs are mostly in the first part of the run. After that, training stays above validation. At epoch `120` the train-minus-validation gap is `5.01`, `5.10`, and `5.61` percentage points.

Official test at epoch `10` still differs by seed. Seed `112` is `63.35%` and seed `111` is `58.32%`, a gap of `5.03` percentage points. By epoch `120` that gap is `0.25` percentage points.

The markers keep rising after the learning rate has fallen below `0.02`, and the steps get small. From epoch `100` to epoch `120` the official-test gains are `0.56`, `0.35`, and `0.72` percentage points. Over those same epochs the parameter count goes from `1,517,568` to `1,813,248`.

Seed `110` validation peaks at epoch `116` at `91.98%`, then ends at `91.60%`, while its official test is still highest at epoch `120`. Seed `112` validation peaks at epoch `108` at `91.02%` and ends at `90.76%`. Seed `111` validation is highest at epoch `120`.

### What did search insert?

`history.pt` stores one row per executed action. Validation accuracy is measured on the `5,000` image slice immediately before and after `action.execute`. The next generation's official test is the later window, not this immediate change.

Every executed action is `AddSeqConvLayer`. There are `11` of them, at generations `0` through `10`. Generation `11` has no action. The final `state_dict` contains `seq_conv_0` through `seq_conv_10`. Phase 1 `history.pt` does not store which layers the new conv was tied to. The edges are in the next section.

The immediate validation change is at most `0.08` percentage points. The largest is seed `110`, generation `0`, from `58.80%` to `58.88%`. The other immediate changes are `0.04` percentage points or smaller. The jump in official test from generation `0` to generation `1` is `17.50`, `17.16`, and `9.35` percentage points. That jump is the next `10` epochs, not the immediate insert.

`param_count` is recorded after the generation's SGD and before the action, then copied onto every epoch of that generation. It stays `187,008` through epoch `10`. Each later generation is `147,840` parameters higher. Epoch `120` is `1,813,248` on every seed. `147,840` is one `128`-channel `3×3` convolution, its bias, and one `BatchNorm2d` of `128` channels.

### Were the phase 1 actions the same?

The action name is the same on every seed and every generation: `AddSeqConvLayer`. That is the only legal action, so the name cannot differ. The insert edge is the same for generations `0` through `3` and different after that.

The edges below are the `growingnn.log` lines from `2026-10-14 08:07:32` to `09:41:23`. Each line's validation accuracy matches `action_val_acc_before` in that seed's `history.pt`. Phase 1 `history.pt` stores the action name and those accuracies. It does not store the edge.

| Generation | Seed 110 | Seed 111 | Seed 112 |
| --- | --- | --- | --- |
| 0 | `conv2` to `linear` | `conv2` to `linear` | `conv2` to `linear` |
| 1 | `seq_conv_0` to `linear` | `seq_conv_0` to `linear` | `seq_conv_0` to `linear` |
| 2 | `seq_conv_1` to `linear` | `seq_conv_1` to `linear` | `seq_conv_1` to `linear` |
| 3 | `seq_conv_2` to `linear` | `seq_conv_2` to `linear` | `seq_conv_2` to `linear` |
| 4 | `seq_conv_2` to `seq_conv_3` | `seq_conv_1` to `seq_conv_2` | `seq_conv_3` to `linear` |
| 5 | `seq_conv_0` to `seq_conv_1` | `seq_conv_3` to `linear` | `seq_conv_3` to `seq_conv_4` |
| 6 | `seq_conv_4` to `seq_conv_3` | `seq_conv_4` to `seq_conv_2` | `seq_conv_4` to `linear` |
| 7 | `seq_conv_4` to `seq_conv_6` | `seq_conv_5` to `linear` | `seq_conv_3` to `seq_conv_5` |
| 8 | `seq_conv_5` to `seq_conv_1` | `seq_conv_0` to `seq_conv_1` | `seq_conv_6` to `linear` |
| 9 | `seq_conv_7` to `seq_conv_6` | `seq_conv_6` to `seq_conv_2` | `seq_conv_0` to `seq_conv_1` |
| 10 | `seq_conv_0` to `seq_conv_5` | `seq_conv_1` to `seq_conv_4` | `seq_conv_0` to `seq_conv_9` |

Four of the eleven edges match on all three seeds. Those are generations `0` through `3`: `conv2` to `linear`, then the newest convolution toward `linear`. The other seven edges do not match. The parameter step is still `147,840` on every seed, because every legal sequential convolution in this starter has the same shape.

### Conclusions

`train_generations` with this recipe reaches a last official test of `91.11%`, `91.17%`, and `91.36%`. The best saved official test is generation `11` for every seed. The same runs leave training accuracy near `96.5%` and validation near `91.2%`.

Another same-width sequential convolution late in the cosine adds `147,840` parameters and a fraction of a percentage point of official test. From epoch `100` to epoch `120` two such convolutions add `295,680` parameters and at most `0.72` percentage points of official test.

### How seed 112 grew

Seed `112` is the best seed in this run. Its last official test is `91.36%`, the highest of the three. The graphs below replay the `11` `AddSeqConvLayer` edges logged for that seed. The log lines are in `growingnn.log` from `2026-10-14 09:15:07` to `09:41:23`. Each line's validation accuracy matches `history.pt` for seed `112`.

![Seed 112 at generation 0](/assets/experiments/008-architecture-seed112-start.png)

> [!CAPTION] Figure 4. Filtered FX graph of FourLayerCifarNet at generation 0, before any action. Boxes are modules and functions. Convolution labels show the weight shape.

![Seed 112 after generation 3](/assets/experiments/008-architecture-seed112-gen3.png)

> [!CAPTION] Figure 5. Filtered FX graph of seed 112 after the generation 3 action. Four sequential convolutions, each with BatchNorm and ReLU, have been inserted before the global pool.

![Seed 112 after generation 10](/assets/experiments/008-architecture-seed112-final.png)

> [!CAPTION] Figure 6. Filtered FX graph of seed 112 after the generation 10 action. Eleven sequential convolutions are in the chain. Generation 11 adds no layer.

The logged edges, in order, are `conv2` to `linear`, `seq_conv_0` to `linear`, `seq_conv_1` to `linear`, `seq_conv_2` to `linear`, `seq_conv_3` to `linear`, `seq_conv_3` to `seq_conv_4`, `seq_conv_4` to `linear`, `seq_conv_3` to `seq_conv_5`, `seq_conv_6` to `linear`, `seq_conv_0` to `seq_conv_1`, and `seq_conv_0` to `seq_conv_9`.

## Phase 2: which actions search can choose

Phase 2 uses the phase 1 starter, the same `12` generations of `10` epochs, the same cosine, the same cutout, the same label smoothing, the same momentum, weight decay, and gradient clip, and the same validation split. The only change is the set of actions Monte Carlo is allowed to return. Each group trains seeds `110`, `111`, and `112`. The comparison group is only convolution, the phase 1 run, where search can add only `AddSeqConvLayer`.

### The three phase 2 groups

| Group | Actions search may choose |
| --- | --- |
| `all` | Sequential linear, residual linear, sequential conv, residual conv, dropout `0.1` / `0.2` / `0.5`, add neurons `1.1` / `1.5` / `2.0`, delete layer, and delete neurons `0.1` / `0.5` / `0.9`. |
| `grow` | Sequential linear, residual linear, sequential conv, residual conv, and add neurons `1.1` / `1.5` / `2.0`. Deletes and dropout stay off. |
| `seq_res` | Sequential linear, residual linear, sequential conv, and residual conv. Add neurons, deletes, and dropout stay off. |

```
pipenv run python -u experiments/train_cifar10_exp008_generation_only.py --phase 2
```

Output: `experiments/output/train_cifar10/runs/exp008_generation_only_phase2/<group>/seed_<seed>/`

Each phase 2 `history.pt` stores `action_name`, `action_from`, and `action_to`. All three groups have those files for seeds `110`, `111`, and `112`.

| Group | Seed `110` | Seed `111` | Seed `112` | Mean |
| --- | ---: | ---: | ---: | ---: |
| Only convolution | `91.11%` | `91.17%` | `91.36%` | `91.21%` |
| `all` | `85.70%` | `90.40%` | `80.94%` | `85.68%` |
| `grow` | `85.54%` | `90.81%` | `82.29%` | `86.21%` |
| `seq_res` | `81.52%` | `83.06%` | `86.46%` | `83.68%` |

Every later mean is below the only convolution mean. `grow` is the highest of the three, `5.00` percentage points below only convolution. `all` is `5.53` percentage points below. `seq_res` is `7.53` percentage points below. The closest single seed is `grow` seed `111`, at `90.81%`, which is `0.36` percentage points below that seed in the only convolution group.

![Mean final accuracy by action group](/assets/experiments/008-phase2-group-means.png)

> [!CAPTION] Figure 7. Mean accuracy at epoch 120, in percent. The clusters, from left to right, are only convolution, all, grow, and sequential and residual. Blue is mean training accuracy, green is mean official test accuracy, and orange is mean validation accuracy. Black dots are seeds 110, 111, and 112. The dashed line is 90%. The vertical axis starts at 75%.

The official-test means are `91.21%`, `85.68%`, `86.21%`, and `83.68%`. The training means are `96.45%`, `89.16%`, `90.22%`, and `87.25%`. The validation means are `91.21%`, `85.87%`, `85.59%`, and `83.44%`. The black dots show that those means hide a wide seed spread in every group except only convolution.

![Train, official test, and validation for four action groups](/assets/experiments/008-phase2-mode-grid.png)

> [!CAPTION] Figure 8. Four rows are the action groups: only convolution, all, grow, and sequential and residual. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

Only convolution is the phase 1 run. Search could add only `AddSeqConvLayer`. Its three seeds stay close. The other three rows do not. Seed `110` in `all` is the line that falls to about `10%` near epoch `41` and then recovers. Seed `111` in `grow` is the line that falls near epoch `41` and then climbs past `90%` official test.

![Parameter count for four action groups](/assets/experiments/008-phase2-param-grid.png)

> [!CAPTION] Figure 9. Four panels are the action groups: only convolution, all, grow, and sequential and residual. Each line is the parameter count of one seed, in millions. Blue is seed 110, orange is seed 111, and green is seed 112. Every panel uses the same vertical axis, starting at 0. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count.

Only convolution is one staircase on every seed, ending at `1,813,248` parameters, so the three lines overlap. The other three panels separate. `grow` seed `110` ends highest, at `3,865,728`. Sequential and residual stays near one million: `959,744`, `1,140,096`, and `1,041,024`. `all` seed `110` ends at `794,880`, below the only convolution staircase.

### Did the all group match only convolution?

Official test accuracy is the fraction correct on the `10,000` image test set. Dashed lines are the only convolution group. Solid lines are the `all` group. A color is one seed.

![Official test, phase 1 and phase 2 all](/assets/experiments/008-phase2-official-test.png)

> [!CAPTION] Figure 10. Official test accuracy in percent. Dashed lines are the only convolution group. Solid lines with markers are the phase 2 all group. Blue is seed 110, orange is seed 111, and green is seed 112. The dotted line is 90%. The vertical axis starts at 50%.

| Seed | Only convolution last official test | All last official test | All minus only convolution |
| --- | ---: | ---: | ---: |
| `110` | `91.11%` | `85.70%` | `-5.41` percentage points |
| `111` | `91.17%` | `90.40%` | `-0.77` percentage points |
| `112` | `91.36%` | `80.94%` | `-10.42` percentage points |
| Mean | `91.21%` | `85.68%` | `-5.53` percentage points |

The `all` group mean is `5.53` percentage points below the only convolution mean. Seed `111` is the only `all` seed at or above `90%`. Its last official test is still below that seed in the only convolution group. Every saved official-test series in the `all` group is highest at generation `11`.

| Generation | Epoch | Seed `110` | Seed `111` | Seed `112` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `10` | `56.49%` | `57.76%` | `61.83%` |
| `1` | `20` | `65.79%` | `76.89%` | `59.53%` |
| `2` | `30` | `75.17%` | `78.81%` | `61.57%` |
| `3` | `40` | `73.95%` | `80.31%` | `62.19%` |
| `4` | `50` | `71.55%` | `83.46%` | `66.21%` |
| `5` | `60` | `77.22%` | `84.99%` | `67.76%` |
| `6` | `70` | `79.59%` | `87.06%` | `71.75%` |
| `7` | `80` | `80.61%` | `87.59%` | `73.09%` |
| `8` | `90` | `84.08%` | `89.25%` | `75.67%` |
| `9` | `100` | `84.84%` | `89.65%` | `79.62%` |
| `10` | `110` | `85.16%` | `90.25%` | `80.54%` |
| `11` | `120` | `85.70%` | `90.40%` | `80.94%` |

Seed `111` rises at every generation and first reaches `90%` at generation `10`. Seed `110` falls from generation `2` to generation `3` by `1.22` percentage points, then from generation `3` to generation `4` by `2.40` percentage points, and then rises. Seed `112` falls from generation `0` to generation `1` by `2.30` percentage points, and then rises. The last official-test values span `9.46` percentage points, from `80.94%` to `90.40%`. Only convolution spanned `0.25` percentage points.

| Seed | Last train | Last validation | Last official test | End parameters |
| --- | ---: | ---: | ---: | ---: |
| `110` | `89.81%` | `86.20%` | `85.70%` | `794,880` |
| `111` | `95.25%` | `90.50%` | `90.40%` | `3,703,651` |
| `112` | `82.42%` | `80.90%` | `80.94%` | `1,041,487` |

Seed `112` training accuracy ends at `82.42%`. Only convolution training accuracy ended near `96.5%` on every seed. The low official test on seed `112` sits next to a low training accuracy, not next to a large train-minus-validation gap. That gap at epoch `120` is `3.61`, `4.75`, and `1.52` percentage points.

![Parameter count, phase 2 all](/assets/experiments/008-phase2-parameters.png)

> [!CAPTION] Figure 11. Parameter count by epoch for the phase 2 all group. Blue is seed 110, orange is seed 111, and green is seed 112. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count.

Only convolution ended at `1,813,248` parameters on every seed. The `all` group does not. Seed `111` ends at `3,703,651`. Seed `110` ends at `794,880`. Seed `112` ends at `1,041,487`. Seed `112` generation `2` executes `AddSeqDropoutLayer`. The parameter count stays `547,968` through the next generation.

### How train, test, and validation move

Figure 8 is that comparison. Each row is one action group. Training and validation are one point per epoch. Official test markers are the `12` generation ends.

In the `all` row, seed `112` is the low curve on all three columns after the first generations, and it ends with training, validation, and official test within `1.52` percentage points of each other. Seeds `110` and `111` finish with training above validation. Seed `111` official test rises at every marker and is the only `all` series that crosses `90%`.

Seed `110` training and validation collapse at epoch `41`, the first epoch of generation `4`. Training falls from `75.31%` to `14.29%`, then sits at `10.04%` from epoch `42` through epoch `48`. Validation falls from `73.38%` to `9.66%` and stays near `9.6%` until epoch `49`. Epoch `50` is back to `51.74%` training and `71.10%` validation. Official test is not measured inside that window. It is `73.95%` at epoch `40` and `71.55%` at epoch `50`. Generation `3` had just executed `AddResConvLayer` from `conv1` to `seq_linear_0`. The immediate validation change of that action was `+0.04` percentage points.

In the `grow` row, seed `112` starts as the high official-test marker and ends as the low one, at `82.29%`. Seed `110` official test rises at every marker and ends at `85.54%`. Seed `111` official test falls from `82.89%` at epoch `40` to `70.69%` at epoch `50`. Training at epoch `41` falls from `82.87%` to `50.73%`, then is `77.29%` at epoch `50`. Generation `3` had just executed `AddResConvLayer` from `conv1` to `linear`. Official test passes the pre-drop value at epoch `70`, where it is `85.89%`. Generation `10` is `90.82%` and generation `11` is `90.81%`. This is the only phase 2 seed whose highest official test is not the last generation.

In the sequential and residual row, seed `112` ends highest, at `86.46%` official test and `92.40%` training. Seed `111` official test rises at every marker and ends at `83.06%`. Seed `110` official test falls from `67.28%` at epoch `20` to `57.79%` at epoch `40`, then rises, and ends at `81.52%`. None of the three official-test series reaches `90%`.

Seed `112` training falls from `66.46%` at epoch `20` to `51.00%` at epoch `21`. Seed `110` training falls from `78.43%` at epoch `80` to `62.98%` at epoch `81`. Official test is not measured on those epochs.

### Were the all-group actions the same?

No. The action name is not the same across the three seeds, and the edge is not the same either. Generation `8` is the only generation where all three seeds execute the same action name, `AddResLinearLayer`, and the three edges still differ. No generation has the same action name and the same edge on all three seeds.

Seeds `110` and `111` share one edge: generation `0`, `conv2` to `linear`. The action names differ. Seed `110` executes `AddResConvLayer`. Seed `111` executes `AddSeqConvLayer`. Seeds `111` and `112` share one action and one edge: generation `5`, `AddResLinearLayer` from `linear` to `seq_linear_0`. Seeds `110` and `112` share no edge. Out of `11` generations, the action name matches on `1` generation for seeds `110` and `111`, on `2` generations for seeds `110` and `112`, and on `4` generations for seeds `111` and `112`.

| Generation | Seed `110` | Seed `111` | Seed `112` |
| --- | --- | --- | --- |
| `0` | `AddResConvLayer`, `conv2` to `linear` | `AddSeqConvLayer`, `conv2` to `linear` | `AddSeqLinearLayer`, `linear` to `linear2` |
| `1` | `AddSeqLinearLayer`, `conv2` to `linear` | `AddNeurons`, `linear` ratio `2.0` | `AddResConvLayer`, `conv1` to `seq_linear_0` |
| `2` | `AddResLinearLayer`, `seq_linear_0` to `linear` | `AddResConvLayer`, `seq_conv_0` to `linear` | `AddSeqDropoutLayer`, `seq_linear_0` to `linear2` |
| `3` | `AddResConvLayer`, `conv1` to `seq_linear_0` | `AddSeqLinearLayer`, `linear` to `linear2` | `AddSeqLinearLayer`, `seq_linear_0` to `linear2` |
| `4` | `AddSeqLinearLayer`, `conv2` to `seq_linear_0` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddNeurons`, `seq_linear_1` ratio `1.1` |
| `5` | `AddSeqLinearLayer`, `seq_linear_1` to `seq_linear_0` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddResLinearLayer`, `linear` to `seq_linear_0` |
| `6` | `AddResLinearLayer`, `seq_linear_1` to `seq_linear_2` | `AddResConvLayer`, `conv1` to `seq_linear_0` | `AddSeqLinearLayer`, `seq_linear_1` to `linear2` |
| `7` | `AddSeqLinearLayer`, `seq_linear_1` to `seq_linear_2` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_1` | `AddResLinearLayer`, `seq_linear_0` to `seq_linear_2` |
| `8` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_ZERO_1` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_0` | `AddResLinearLayer`, `linear` to `seq_linear_1` |
| `9` | `AddResLinearLayer`, `seq_linear_1` to `seq_linear_0` | `AddResConvLayer`, `conv1` to `res_linear_EYE_1` | `AddResLinearLayer`, `linear` to `res_linear_EYE_0` |
| `10` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_EYE_1` | `AddNeurons`, `res_linear_EYE_0` ratio `1.1` | `AddSeqLinearLayer`, `linear` to `res_linear_EYE_2` |

Seed `110` executes `5` `AddResLinearLayer`, `4` `AddSeqLinearLayer`, and `2` `AddResConvLayer`. Seed `111` executes `4` `AddResLinearLayer`, `3` `AddResConvLayer`, `2` `AddNeurons`, `1` `AddSeqConvLayer`, and `1` `AddSeqLinearLayer`. Seed `112` executes `4` `AddSeqLinearLayer`, `4` `AddResLinearLayer`, `1` `AddResConvLayer`, `1` `AddSeqDropoutLayer`, and `1` `AddNeurons`. No seed executes `DelLayer` or `DelNeurons`.

The immediate validation change is usually at most a few tenths of a percentage point. The largest is seed `110`, generation `8`, `AddResLinearLayer` from `seq_linear_1` to `res_linear_ZERO_1`: validation goes from `83.56%` to `82.02%`, a drop of `1.54` percentage points. The official-test drops on seeds `110` and `112` are between generation ends. They are the next `10` epochs, not that immediate measurement.

### Did the grow seeds choose the same actions?

No. No generation has the same action name on all three seeds. No pair executes the same action on the same edge at the same generation.

| Seed | Last train | Last validation | Last official test | End parameters |
| --- | ---: | ---: | ---: | ---: |
| `110` | `90.02%` | `84.66%` | `85.54%` | `3,865,728` |
| `111` | `96.15%` | `90.64%` | `90.81%` | `3,634,330` |
| `112` | `84.50%` | `81.48%` | `82.29%` | `943,232` |

| Generation | Epoch | Seed `110` | Seed `111` | Seed `112` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `10` | `56.18%` | `58.27%` | `61.21%` |
| `1` | `20` | `65.83%` | `74.26%` | `64.35%` |
| `2` | `30` | `69.33%` | `79.30%` | `63.29%` |
| `3` | `40` | `69.35%` | `82.89%` | `69.24%` |
| `4` | `50` | `70.06%` | `70.69%` | `66.87%` |
| `5` | `60` | `73.31%` | `80.49%` | `67.73%` |
| `6` | `70` | `76.42%` | `85.89%` | `74.31%` |
| `7` | `80` | `77.25%` | `88.52%` | `76.73%` |
| `8` | `90` | `81.70%` | `89.41%` | `77.67%` |
| `9` | `100` | `84.40%` | `90.51%` | `80.73%` |
| `10` | `110` | `85.08%` | `90.82%` | `81.92%` |
| `11` | `120` | `85.54%` | `90.81%` | `82.29%` |

| Generation | Seed `110` | Seed `111` | Seed `112` |
| --- | --- | --- | --- |
| `0` | `AddNeurons`, `linear` ratio `2.0` | `AddSeqConvLayer`, `conv2` to `linear` | `AddSeqLinearLayer`, `linear` to `linear2` |
| `1` | `AddSeqLinearLayer`, `linear` to `linear2` | `AddNeurons`, `linear` ratio `1.5` | `AddResLinearLayer`, `linear` to `seq_linear_0` |
| `2` | `AddSeqLinearLayer`, `seq_linear_0` to `linear2` | `AddNeurons`, `linear` ratio `1.1` | `AddSeqLinearLayer`, `conv2` to `linear` |
| `3` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddResConvLayer`, `conv1` to `linear` | `AddResConvLayer`, `conv1` to `res_linear_EYE_0` |
| `4` | `AddNeurons`, `seq_linear_0` ratio `1.5` | `AddResConvLayer`, `seq_conv_0` to `linear` | `AddSeqLinearLayer`, `conv1` to `linear2` |
| `5` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_0` | `AddResConvLayer`, `conv2` to `linear` | `AddResLinearLayer`, `seq_linear_0` to `seq_linear_2` |
| `6` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_0` | `AddResConvLayer`, `seq_conv_0` to `linear` | `AddSeqLinearLayer`, `seq_linear_1` to `linear` |
| `7` | `AddResConvLayer`, `conv2` to `res_linear_ZERO_0` | `AddResConvLayer`, `seq_conv_0` to `linear` | `AddResLinearLayer`, `seq_linear_0` to `res_linear_ZERO_0` |
| `8` | `AddResLinearLayer`, `linear` to `seq_linear_1` | `AddResConvLayer`, `conv2` to `linear` | `AddResLinearLayer`, `seq_linear_1` to `linear` |
| `9` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_2` | `AddSeqLinearLayer`, `conv2` to `linear2` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_EYE_0` |
| `10` | `AddSeqLinearLayer`, `linear` to `res_linear_ZERO_0` | `AddSeqLinearLayer`, `seq_linear_0` to `linear2` | `AddResLinearLayer`, `seq_linear_1` to `linear` |

Seed `110` executes `5` `AddResLinearLayer`, `3` `AddSeqLinearLayer`, `2` `AddNeurons`, and `1` `AddResConvLayer`. Seed `111` executes `6` `AddResConvLayer`, `2` `AddNeurons`, `2` `AddSeqLinearLayer`, and `1` `AddSeqConvLayer`. Seed `112` executes `6` `AddResLinearLayer`, `4` `AddSeqLinearLayer`, and `1` `AddResConvLayer`.

### Did the seq_res seeds choose the same actions?

Generation `0` is the same action and the same edge on all three seeds: `AddSeqLinearLayer` from `linear` to `linear2`. That is the only generation with the same edge. Generations `5`, `9`, and `10` share the action name `AddResLinearLayer`, and the edges differ. No seed executes `AddSeqConvLayer`.

| Seed | Last train | Last validation | Last official test | End parameters |
| --- | ---: | ---: | ---: | ---: |
| `110` | `82.85%` | `81.34%` | `81.52%` | `959,744` |
| `111` | `86.49%` | `82.78%` | `83.06%` | `1,140,096` |
| `112` | `92.40%` | `86.20%` | `86.46%` | `1,041,024` |

| Generation | Epoch | Seed `110` | Seed `111` | Seed `112` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `10` | `55.13%` | `57.47%` | `61.23%` |
| `1` | `20` | `67.28%` | `66.34%` | `65.46%` |
| `2` | `30` | `64.82%` | `67.66%` | `69.65%` |
| `3` | `40` | `57.79%` | `68.87%` | `73.37%` |
| `4` | `50` | `67.79%` | `73.62%` | `79.23%` |
| `5` | `60` | `70.76%` | `74.51%` | `82.12%` |
| `6` | `70` | `74.43%` | `76.34%` | `78.62%` |
| `7` | `80` | `76.26%` | `77.28%` | `83.19%` |
| `8` | `90` | `77.33%` | `79.02%` | `85.05%` |
| `9` | `100` | `79.23%` | `82.21%` | `85.90%` |
| `10` | `110` | `81.24%` | `82.92%` | `86.23%` |
| `11` | `120` | `81.52%` | `83.06%` | `86.46%` |

| Generation | Seed `110` | Seed `111` | Seed `112` |
| --- | --- | --- | --- |
| `0` | `AddSeqLinearLayer`, `linear` to `linear2` | `AddSeqLinearLayer`, `linear` to `linear2` | `AddSeqLinearLayer`, `linear` to `linear2` |
| `1` | `AddSeqLinearLayer`, `conv2` to `linear` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddResConvLayer`, `conv2` to `linear` |
| `2` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddSeqLinearLayer`, `seq_linear_0` to `linear2` | `AddSeqLinearLayer`, `conv2` to `linear` |
| `3` | `AddResLinearLayer`, `linear` to `seq_linear_0` | `AddResLinearLayer`, `linear` to `seq_linear_1` | `AddResConvLayer`, `conv2` to `linear` |
| `4` | `AddResLinearLayer`, `seq_linear_1` to `linear` | `AddResLinearLayer`, `linear` to `res_linear_EYE_0` | `AddSeqLinearLayer`, `seq_linear_1` to `linear` |
| `5` | `AddResLinearLayer`, `res_linear_EYE_1` to `res_linear_ZERO_0` | `AddResLinearLayer`, `linear` to `res_linear_EYE_0` | `AddResLinearLayer`, `seq_linear_1` to `seq_linear_2` |
| `6` | `AddSeqLinearLayer`, `res_linear_EYE_1` to `res_linear_ZERO_0` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_1` | `AddResLinearLayer`, `linear` to `seq_linear_0` |
| `7` | `AddResConvLayer`, `conv1` to `linear` | `AddSeqLinearLayer`, `linear` to `res_linear_ZERO_1` | `AddResLinearLayer`, `res_linear_ZERO_0` to `res_linear_ZERO_1` |
| `8` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_EYE_2` | `AddResConvLayer`, `conv2` to `res_linear_EYE_0` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_ZERO_0` |
| `9` | `AddResLinearLayer`, `seq_linear_1` to `seq_linear_2` | `AddResLinearLayer`, `linear` to `res_linear_ZERO_2` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_ZERO_2` |
| `10` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_EYE_4` | `AddResLinearLayer`, `linear` to `res_linear_EYE_2` | `AddResLinearLayer`, `seq_linear_1` to `res_linear_ZERO_3` |

Seed `110` executes `7` `AddResLinearLayer`, `3` `AddSeqLinearLayer`, and `1` `AddResConvLayer`. Seed `111` executes the same counts. Seed `112` executes `6` `AddResLinearLayer`, `3` `AddSeqLinearLayer`, and `2` `AddResConvLayer`. The counts match more closely than the edges do.

The `history.pt` save times, from the filesystem, run from `2026-10-14 23:23:24` for `all` seed `110` to `2026-10-15 15:26:25` for `seq_res` seed `112`.

### Conclusions

None of the three action lists repeats the only convolution official-test mean of `91.21%`. The `grow` mean is `86.21%`. The `all` mean is `85.68%`. The `seq_res` mean is `83.68%`.

The seeds inside a group do not execute the same actions. In `all`, one generation shares an action name and none shares an edge. In `grow`, no generation shares an action name. In `seq_res`, generation `0` is the same action and the same edge, and three later generations share only the name `AddResLinearLayer`.

### What is still open

A later run that wants official test above the only convolution band needs a different growth step than the action lists tried here. The closest saved seed is `grow` seed `111` at `90.81%`. Phase 3 keeps the all-action list and lengthens the simulation. Its last-official-test mean is `87.61%`, still below only convolution. Experiment 007 can stay as the finish-stage comparison. This page does not replace it.

## Phase 3: how long search looks

Phase 3 asks whether a longer look changes the all-action result. The action list is the phase 2 `all` group. The starter, the `12` generations of `10` epochs, the cosine, the cutout, the label smoothing, the momentum, the weight decay, the gradient clip, and the validation split stay as they are in phase 1. Seeds are `110`, `111`, and `112`. Search still runs after every generation except the last. The score is still accuracy weight `1` and parameter-count weight `0`. The algorithm is still Monte Carlo.

The only change is the simulation budget.

| Run | Actions | Simulation time | Simulation epochs | `simulation_set_size` | Last official-test mean |
| --- | --- | ---: | ---: | ---: | ---: |
| Only convolution | `AddSeqConvLayer` only | `8` s | `1` | `128` | `91.21%` |
| All, short | every structure action | `8` s | `1` | `128` | `85.68%` |
| All, long | every structure action | `100` s | `8` | `2000` | `87.61%` |

`sample_loaders` draws `simulation_set_size` training images and one quarter of that count from validation. The short budget is `128` training images and `32` validation images. The long budget is `2,000` training images and `500` validation images. Each candidate is then trained for the simulation-epoch count.

The three `history.pt` files store `simulation_time` `100.0`, `simulation_epochs` `8`, and `simulation_set_size` `2000`. The long mean is `1.93` percentage points above the short all mean and `3.60` percentage points below the only convolution mean. Two seeds rise relative to their short-all result. Seed `111` falls.

| Seed | Only convolution | All, short | All, long | Long minus short | Long minus only convolution |
| --- | ---: | ---: | ---: | ---: | ---: |
| `110` | `91.11%` | `85.70%` | `88.45%` | `+2.75` percentage points | `-2.66` percentage points |
| `111` | `91.17%` | `90.40%` | `85.04%` | `-5.36` percentage points | `-6.13` percentage points |
| `112` | `91.36%` | `80.94%` | `89.35%` | `+8.41` percentage points | `-2.01` percentage points |
| Mean | `91.21%` | `85.68%` | `87.61%` | `+1.93` percentage points | `-3.60` percentage points |

### Did the longer look change official test?

Official test accuracy is the fraction correct on the `10,000` image test set. Figure 12 puts only convolution, the short all group, and the long all group on one grid. The middle column is official test.

![Train, official test, and validation for three search budgets](/assets/experiments/008-phase3-mode-grid.png)

> [!CAPTION] Figure 12. Three rows are only convolution, the short all group, and the long all group. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

| Generation | Epoch | Seed `110` | Seed `111` | Seed `112` |
| --- | ---: | ---: | ---: | ---: |
| `0` | `10` | `59.90%` | `56.17%` | `60.91%` |
| `1` | `20` | `62.18%` | `61.66%` | `64.52%` |
| `2` | `30` | `63.06%` | `63.19%` | `65.36%` |
| `3` | `40` | `75.68%` | `68.99%` | `70.21%` |
| `4` | `50` | `82.04%` | `69.38%` | `73.94%` |
| `5` | `60` | `83.90%` | `69.16%` | `80.85%` |
| `6` | `70` | `84.07%` | `72.69%` | `85.08%` |
| `7` | `80` | `86.00%` | `75.97%` | `86.25%` |
| `8` | `90` | `86.90%` | `80.54%` | `87.88%` |
| `9` | `100` | `87.82%` | `82.08%` | `88.96%` |
| `10` | `110` | `88.05%` | `84.38%` | `89.15%` |
| `11` | `120` | `88.45%` | `85.04%` | `89.35%` |

Seeds `110` and `112` rise at every generation. Seed `111` falls once, from `69.38%` at generation `4` to `69.16%` at generation `5`, a drop of `0.22` percentage points, and then rises. Every series is highest at generation `11`. None of the `12` points reaches `90%`. The last values span `4.31` percentage points, from `85.04%` to `89.35%`.

Seed `112` is the closest long-run seed to only convolution, `2.01` percentage points below that seed's phase 1 result. Seed `111` was the closest short-all seed, at `90.40%`, and it is the long-run seed that ends lowest.

### How do train, test, and validation move?

Training and validation are one point per epoch. Official test markers are the `12` generation ends. Those curves are the three columns of Figure 12. The long all group is the bottom row.

| Seed | Last train | Last validation | Last official test | End parameters |
| --- | ---: | ---: | ---: | ---: |
| `110` | `91.03%` | `88.64%` | `88.45%` | `1,157,896` |
| `111` | `82.24%` | `84.68%` | `85.04%` | `681,011` |
| `112` | `90.23%` | `89.66%` | `89.35%` | `1,356,388` |
| Mean | `87.83%` | `87.66%` | `87.61%` | |

Seed `111` ends with validation `2.44` percentage points above training, and with official test above training. Seeds `110` and `112` end with training above validation. The train-minus-validation gaps are `2.39`, `-2.44`, and `0.57` percentage points.

Seed `111` training falls from `67.14%` at epoch `40` to `61.71%` at epoch `41`, and from `78.91%` at epoch `90` to `71.54%` at epoch `91`. Validation at epoch `41` is `68.10%`. Validation at epoch `91` is `82.66%`, above the epoch `90` value of `80.50%`. Official test is `68.99%` at epoch `40` and `69.38%` at epoch `50`. It is `80.54%` at epoch `90` and `82.08%` at epoch `100`. Seeds `110` and `112` have no training drop larger than `5` percentage points.

Generation `3` on seed `111` executed `DelNeurons` on `linear` with ratio `0.5` after the epoch `40` measurement. The immediate validation change of that action is `69.04%` to `55.30%`, a drop of `13.74` percentage points. The official-test values above are the generation ends, not that immediate measurement.

### How did the parameter count move?

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. A delete shows up as a step down at the first epoch of the next generation. Dropout does not change the count.

![Parameter count, long all](/assets/experiments/008-phase3-parameters.png)

> [!CAPTION] Figure 13. Parameter count by epoch for the long all group. Blue is seed 110, orange is seed 111, and green is seed 112. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count.

Only convolution ended at `1,813,248` parameters on every seed. All three long runs end below that: `1,157,896`, `681,011`, and `1,356,388`. Seed `111` is the low line. Four of its actions are `DelNeurons`. The short all seed `111` ended at `3,703,651` parameters. These are different searches, not the same actions measured twice.

### Were the long-simulation actions the same?

No. No generation has the same action name on all three seeds. Seeds `111` and `112` share two full actions: generation `0`, `AddSeqDropoutLayer` from `linear` to `linear2`, and generation `1`, `DelNeurons` on `linear` with ratio `0.5`. Generation `10` shares the name `AddResConvLayer` on those two seeds, and the edges differ. Out of `11` generations, the action name matches on `2` generations for seeds `110` and `111`, on `1` generation for seeds `110` and `112`, and on `3` generations for seeds `111` and `112`.

| Generation | Seed `110` | Seed `111` | Seed `112` |
| --- | --- | --- | --- |
| `0` | `AddSeqLinearLayer`, `conv2` to `linear` | `AddSeqDropoutLayer`, `linear` to `linear2` | `AddSeqDropoutLayer`, `linear` to `linear2` |
| `1` | `AddNeurons`, `seq_linear_0` ratio `2.0` | `DelNeurons`, `linear` ratio `0.5` | `DelNeurons`, `linear` ratio `0.5` |
| `2` | `AddSeqConvLayer`, `conv2` to `seq_linear_0` | `AddSeqLinearLayer`, `linear` to `linear2` | `AddNeurons`, `linear` ratio `1.5` |
| `3` | `DelNeurons`, `seq_linear_0` ratio `0.5` | `DelNeurons`, `linear` ratio `0.5` | `AddNeurons`, `linear` ratio `2.0` |
| `4` | `AddNeurons`, `seq_linear_0` ratio `1.5` | `DelNeurons`, `seq_linear_0` ratio `0.5` | `AddSeqConvLayer`, `conv2` to `linear` |
| `5` | `AddSeqDropoutLayer`, `seq_linear_0` to `linear` | `DelNeurons`, `seq_linear_0` ratio `0.9` | `AddSeqConvLayer`, `seq_conv_0` to `linear` |
| `6` | `AddNeurons`, `seq_linear_0` ratio `1.1` | `AddSeqDropoutLayer`, `seq_linear_0` to `linear2` | `AddNeurons`, `linear` ratio `2.0` |
| `7` | `AddResConvLayer`, `seq_conv_0` to `seq_linear_0` | `AddSeqConvLayer`, `conv2` to `linear` | `DelNeurons`, `linear` ratio `0.9` |
| `8` | `AddResConvLayer`, `seq_conv_0` to `seq_linear_0` | `AddResConvLayer`, `conv1` to `linear` | `AddSeqDropoutLayer`, `seq_conv_0` to `seq_conv_1` |
| `9` | `AddResConvLayer`, `conv2` to `seq_linear_0` | `AddSeqConvLayer`, `seq_conv_0` to `linear` | `AddSeqLinearLayer`, `seq_conv_1` to `linear` |
| `10` | `AddSeqLinearLayer`, `seq_linear_0` to `linear` | `AddResConvLayer`, `seq_conv_0` to `seq_conv_1` | `AddResConvLayer`, `conv1` to `linear` |

Seed `110` executes `3` `AddNeurons`, `3` `AddResConvLayer`, `2` `AddSeqLinearLayer`, `1` `AddSeqConvLayer`, `1` `DelNeurons`, and `1` `AddSeqDropoutLayer`. Seed `111` executes `4` `DelNeurons`, `2` `AddSeqDropoutLayer`, `2` `AddSeqConvLayer`, `2` `AddResConvLayer`, and `1` `AddSeqLinearLayer`. Seed `112` executes `3` `AddNeurons`, `2` `AddSeqDropoutLayer`, `2` `DelNeurons`, `2` `AddSeqConvLayer`, `1` `AddSeqLinearLayer`, and `1` `AddResConvLayer`. No seed executes `AddResLinearLayer` or `DelLayer`.

The short all group executed `AddResLinearLayer` on every seed. The long group does not. That is a difference in the saved actions, not a measured cause.

The `history.pt` save times, from the filesystem, are `2026-10-16 00:09` for seed `110`, `01:00` for seed `111`, and `01:53` for seed `112`.

### Conclusions

The longer all-action simulation does not reach the only convolution official-test mean of `91.21%`. Its mean is `87.61%`. That is `1.93` percentage points above the short all mean of `85.68%`, and the gain is not on every seed. Seed `111` is `5.36` percentage points below its own short-all result.

The three long seeds do not execute the same actions. Two early actions match on seeds `111` and `112`. None match on all three seeds. The saved long actions include delete-neurons and dropout, and they include no residual linear add.

## Phase 4: full augmented grade

Phase 4 grades every legal action on the full training set. The simulation training set is all `45,000` augmented images. The simulation validation set is all `5,000` validation images. The compared simulation epoch counts are `1`, `2`, and `4`. Simulation time is `300` seconds. Extra actions inside the score are `0`.

This amount of data is too much. Each search trains every legal action for the full simulation-epoch count. A seed writes `history.pt` only when its `12` generations finish. The `2` epoch and `4` epoch grades have no `history.pt`. Those runs were not finished because the `45,000` image grade took too much time.

The `1` epoch grade did finish, for seeds `110`, `111`, and `112`. The files are `experiments/output/train_cifar10/runs/exp008_generation_only_phase4/epochs_1/seed_<seed>/history.pt`. Each file stores `simulation_epochs` `1`, `simulation_set_size` `45000`, and `simulation_augmented` true. The filesystem save times are `2026-10-16 23:11` for seed `110`, `2026-10-17 11:45` for seed `111`, and `2026-10-17 13:44` for seed `112`.

| Seed | Last train | Last validation | Last official test |
| --- | ---: | ---: | ---: |
| `110` | `90.94%` | `88.48%` | `88.58%` |
| `111` | `91.62%` | `88.48%` | `88.15%` |
| `112` | `90.55%` | `87.68%` | `87.57%` |
| Mean | `91.04%` | `88.21%` | `88.10%` |

The finished mean official test is `88.10%`. The phase 1 only-convolution mean is `91.21%`. The `1` epoch grade is `3.11` percentage points below that mean. The `2` epoch and `4` epoch grades are not part of this comparison.

## Phase 5: first 5,000 clean images

Phase 5 asks whether a longer simulation grade changes the actions search chooses. The action list, the Monte Carlo algorithm, the `300` second budget, the `5,000` validation images, and the extra rollout depth of `0` stay as they are in phase 4. The real training stays `12` generations of `10` epochs on the `45,000` augmented images. The only change in the rehearsal is the training slice: `prefix_clean_loader` keeps the first `5,000` images of the clean training loader. That slice has no crop, flip, or cutout. The compared simulation lengths are `1`, `2`, `4`, `5`, `6`, and `7` epochs.

| Setting | Value |
| --- | --- |
| Simulation training images | `5,000`, clean, the first `5,000` |
| Simulation validation images | `5,000` |
| Simulation epochs | `1`, `2`, `4`, `5`, `6`, `7` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Output | `exp008_generation_only_phase5/epochs_<N>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at epoch `120`. All six grades finished, for seeds `110`, `111`, and `112`.

| Seed | `1` epoch | `2` epochs | `4` epochs | `5` epochs | `6` epochs | `7` epochs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `110` | `77.47%` | `81.74%` | `88.85%` | `80.57%` | `86.76%` | `85.82%` |
| `111` | `90.35%` | `82.06%` | `86.42%` | `89.68%` | `85.28%` | `89.75%` |
| `112` | `81.70%` | `80.87%` | `87.80%` | `87.92%` | `81.48%` | `80.85%` |
| Mean | `83.17%` | `81.56%` | `87.69%` | `86.06%` | `84.51%` | `85.47%` |

The `4` epoch mean, `87.69%`, is the highest. The `5` epoch mean is `1.63` percentage points below it. The `6` epoch mean is `3.18` percentage points below it. The `7` epoch mean is `2.22` percentage points below it. The files are `experiments/output/train_cifar10/runs/exp008_generation_only_phase5/epochs_<N>/seed_<seed>/history.pt`. Each file stores `simulation_set_size` `5000`, `simulation_augmented` false, and `simulation_clean_prefix` `5000`. Save times, from the filesystem, run from `2026-10-18 22:38` for the `1` epoch seed `110` to `2026-10-20 21:32` for the `7` epoch seed `112`. `history.pt` stores the chosen action. It does not store the other candidates search graded.

### How did the six grades move?

Figure 14 is one row per simulation length. The middle column is official test.

![Train, official test, and validation for six simulation lengths](/assets/experiments/008-phase5-epoch-grid.png)

> [!CAPTION] Figure 14. Six rows are the 1, 2, 4, 5, 6, and 7 epoch grades on the first 5,000 clean images. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

The `1` epoch last values span `12.88` percentage points. Seed `110` falls from `66.38%` at generation `1` to `50.46%` at generation `2`. The `4` epoch last values span `2.43` percentage points, the smallest span of the six grades. Two series peak one generation before the end. Seed `112` at `5` epochs is `87.99%` at generation `10` and `87.92%` at generation `11`. Seed `110` at `6` epochs is `86.80%` at generation `10` and `86.76%` at generation `11`. Seed `110` at `5` epochs falls from `71.44%` at generation `3` to `64.71%` at generation `4`.

### Which actions did each grade choose?

Figure 15 counts the executed action type. Each seed executed `11` actions in each grade. A missing bar means that seed never chose that type in that grade.

![Chosen action counts by simulation length](/assets/experiments/008-phase5-action-counts.png)

> [!CAPTION] Figure 15. Times each action type was executed. Six panels are the 1, 2, 4, 5, 6, and 7 epoch grades. Each group is one action type. Blue is seed 110, orange is seed 111, and green is seed 112. The vertical axis runs from 0 to 11, the number of searches in a grade.

The `1` epoch grade is the one in which seed `111` chooses `5` residual convolutions and `3` sequential convolutions. The `2` epoch grade chooses no convolution on any seed. The `5` epoch grade ends at `80.57%`, `89.68%`, and `87.92%`. Seed `110` chooses no convolution there, and `3` of its actions delete neurons. The `6` epoch grade ends at `86.76%`, `85.28%`, and `81.48%`. Seed `112` chooses `4` neuron deletes and no convolution. The `7` epoch grade ends at `85.82%`, `89.75%`, and `80.85%`. Seed `112` chooses `5` neuron deletes and ends at `155,153` parameters. Seed `111` chooses `4` neuron adds and `3` convolutions, and ends at `2,042,730` parameters.

Generation `0` of the `1` epoch grade on seed `110` executed `DelNeurons` on `linear` with ratio `0.1` after the epoch `10` measurement. Validation measured on the real model after that action was applied fell from `58.98%` to `23.38%`, a drop of `35.60` percentage points. That number is the committed action, not a simulation score of the candidates that were not chosen.

### How did the parameter count move?

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. A delete shows up as a step down at the first epoch of the next generation. Dropout does not change the count.

![Parameter count by simulation length](/assets/experiments/008-phase5-parameters.png)

> [!CAPTION] Figure 16. Parameter count by epoch. Six panels are the 1, 2, 4, 5, 6, and 7 epoch grades. Blue is seed 110, orange is seed 111, and green is seed 112. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count. Every panel uses the same vertical axis.

The `5` epoch grade ends at `162,475`, `586,520`, and `1,082,585` parameters. The `6` epoch grade ends at `1,758,844`, `818,060`, and `420,977`. The `7` epoch grade ends at `1,378,292`, `2,042,730`, and `155,153`. Seed `112` at `7` epochs is the low line. Five of its actions delete neurons.

### How does the 1 epoch grade sit next to the older recipes?

Figure 17 is context. It is not the six-length comparison. The top row is phase 1, only convolution, mean official test `91.21%`. The middle row is the phase 4 `1` epoch grade on all `45,000` augmented images, mean official test `88.10%`. The bottom row is the phase 5 `1` epoch grade, mean official test `83.17%`. The highest phase 5 mean, the `4` epoch grade at `87.69%`, sits `0.41` percentage points below that phase 4 mean.

![Train, official test, and validation next to the older recipes](/assets/experiments/008-phase5-mode-grid.png)

> [!CAPTION] Figure 17. Three rows are only convolution, the 45,000-image augmented 1-epoch grade, and the 5,000-image clean 1-epoch grade. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

### Conclusions

A longer rehearsal does not keep raising the mean. The means are `83.17%`, `81.56%`, `87.69%`, `86.06%`, `84.51%`, and `85.47%` for `1`, `2`, `4`, `5`, `6`, and `7` simulation epochs. The `4` epoch mean is the highest. The three seeds in that grade end within `2.43` percentage points.

The best single seed is still the `1` epoch seed `111`, at `90.35%`. Seed `111` at `7` epochs ends at `89.75%`. Seed `112` at `7` epochs ends at `80.85%`, after `5` neuron deletes.

The phase 1 only-convolution mean is `91.21%`. The phase 4 `1` epoch mean is `88.10%`. No phase 5 grade reaches either mean.

## Phase 6: smaller clean prefixes

Phase 6 asks whether a smaller simulation training slice changes the actions search chooses. Phase 5 fixed the slice at `5,000` images and found the highest mean at `4` simulation epochs. Phase 6 keeps that length. The action list, the Monte Carlo algorithm, the `300` second budget, the `5,000` validation images, and the extra rollout depth of `0` stay as they are in phase 5. The real training stays `12` generations of `10` epochs on the `45,000` augmented images. The only change is how many clean images `prefix_clean_loader` keeps. Each size is the previous size divided by two, rounded down.

| Setting | Value |
| --- | --- |
| Simulation training images | `5,000`, `2,500`, `1,250`, `625`, `312`, `156`, `78`, clean, from the start of the loader |
| Simulation validation images | `5,000` |
| Simulation epochs | `4` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Output | `exp008_generation_only_phase6/size_<N>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at epoch `120`. All seven grades finished, for seeds `110`, `111`, and `112`.

| Seed | `5,000` | `2,500` | `1,250` | `625` | `312` | `156` | `78` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `110` | `81.84%` | `88.75%` | `90.12%` | `81.67%` | `74.00%` | `88.15%` | `81.46%` |
| `111` | `85.68%` | `88.86%` | `85.85%` | `82.48%` | `81.36%` | `81.20%` | `79.94%` |
| `112` | `89.44%` | `89.41%` | `88.76%` | `89.28%` | `87.72%` | `81.50%` | `88.21%` |
| Mean | `85.65%` | `89.01%` | `88.24%` | `84.48%` | `81.03%` | `83.62%` | `83.20%` |

Figure 18 plots that table. The horizontal axis is the clean prefix size, in the same order as the columns.

![Official test by clean prefix size](/assets/experiments/008-phase6-final-test.png)

> [!CAPTION] Figure 18. Official test accuracy at epoch 120, in percent. The horizontal axis is the number of clean simulation images. The solid line is the mean of the three seeds. Blue is seed 110, orange is seed 111, and green is seed 112. The dotted line is 90%. The vertical axis runs from 70% to 95%.

The `2,500` image mean, `89.01%`, is the highest. It is `1.32` percentage points above the phase 5 `4` epoch mean of `87.69%`. The `312` image mean, `81.03%`, is the lowest. It is `7.98` percentage points below the `2,500` image mean. The `156` image mean is `2.59` percentage points above the `312` image mean. The `78` image mean is `2.17` percentage points above it. The files are `experiments/output/train_cifar10/runs/exp008_generation_only_phase6/size_<N>/seed_<seed>/history.pt`. Each file stores `simulation_epochs` `4`, `simulation_augmented` false, and `simulation_clean_prefix` equal to that size. Save times, from the filesystem, run from `2026-10-21 00:35` for the `5,000` image seed `110` to `2026-10-23 17:23` for the `78` image seed `112`. `history.pt` stores the chosen action. It does not store the other candidates search graded.

The `5,000` image grade uses the same settings as the phase 5 `4` epoch grade. It is a new run. Monte Carlo stops at `300` seconds of wall-clock time, in `montecarlo_alg`, so a later run can grade a different number of candidates. This run ends at `81.84%`, `85.68%`, and `89.44%`. The phase 5 `4` epoch run ended at `88.85%`, `86.42%`, and `87.80%`.

### How did the finished sizes move?

Figure 19 is one row per prefix size. The middle column is official test.

![Train, official test, and validation for seven prefix sizes](/assets/experiments/008-phase6-epoch-grid.png)

> [!CAPTION] Figure 19. Seven rows are the 5,000, 2,500, 1,250, 625, 312, 156, and 78 image grades, each rehearsed for 4 epochs. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

The `5,000` image last values span `7.60` percentage points. Seed `111` is `86.14%` at generation `10` and `85.68%` at generation `11`. The `2,500` image last values span `0.66` percentage points, from `88.75%` to `89.41%`. Seed `110` falls from `80.85%` at generation `2` to `69.59%` at generation `3`, then ends at `88.75%`. The `1,250` image last values span `4.27` percentage points. Seed `110` ends at `90.12%`. Seed `112` falls from `84.20%` at generation `4` to `79.73%` at generation `5`. The `625` image last values span `7.61` percentage points. Seed `111` falls from `58.29%` at generation `0` to `46.17%` at generation `3`, then ends at `82.48%`. Seed `112` is `89.31%` at generation `10` and `89.28%` at generation `11`. The `312` image last values span `13.72` percentage points, from `74.00%` to `87.72%`. Seed `111` is `81.64%` at generation `10` and `81.36%` at generation `11`. The `156` image last values span `6.95` percentage points. Seed `110` falls from `78.86%` at generation `3` to `71.84%` at generation `4`, then ends at `88.15%`. The `78` image last values span `8.27` percentage points, from `79.94%` to `88.21%`.

### Which actions did each finished size choose?

Figure 20 counts the executed action type. Each seed executed `11` actions in each finished grade. A missing bar means that seed never chose that type in that grade.

![Chosen action counts by prefix size](/assets/experiments/008-phase6-action-counts.png)

> [!CAPTION] Figure 20. Times each action type was executed. Seven panels are the 5,000, 2,500, 1,250, 625, 312, 156, and 78 image grades. Each group is one action type. Blue is seed 110, orange is seed 111, and green is seed 112. The vertical axis runs from 0 to 11, the number of searches in a grade.

The `5,000` image seed `110` chooses no convolution and ends at `81.84%`. The `2,500` image grade chooses at least one convolution on every seed. Seed `112` chooses `5` neuron adds and ends at `89.41%`. The `1,250` image seed `110` chooses `2` residual convolutions and `1` sequential convolution, and ends at `90.12%`. The `625` image seeds `110` and `111` choose residual convolutions and no sequential convolution. Seed `112` chooses `2` sequential convolutions and `1` residual convolution, and ends at `89.28%`. The `312` image seed `111` chooses no convolution and ends at `81.36%`. Seed `110` chooses `3` neuron deletes and ends at `74.00%`. The `156` image seed `110` chooses `3` sequential convolutions and `2` residual convolutions, and ends at `88.15%`. Seed `111` chooses no convolution. The `78` image seeds `110` and `111` choose no convolution. Seed `112` chooses `2` sequential convolutions and `3` residual convolutions, and ends at `88.21%`.

### How did the parameter count move?

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. A delete shows up as a step down at the first epoch of the next generation. Dropout does not change the count.

![Parameter count by prefix size](/assets/experiments/008-phase6-parameters.png)

> [!CAPTION] Figure 21. Parameter count by epoch. Seven panels are the 5,000, 2,500, 1,250, 625, 312, 156, and 78 image grades. Blue is seed 110, orange is seed 111, and green is seed 112. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count. Every panel uses the same vertical axis.

The `5,000` image grade ends at `674,968`, `1,142,080`, and `1,330,115` parameters. The `2,500` image grade ends at `684,902`, `1,643,979`, and `1,550,720`. The `1,250` image grade ends at `1,895,355`, `660,871`, and `545,049`. The `625` image grade ends at `2,781,568`, `1,627,326`, and `860,040`. The `312` image grade ends at `279,663`, `943,270`, and `329,949`. The `156` image grade ends at `1,336,192`, `298,856`, and `807,125`. The `78` image grade ends at `279,673`, `260,894`, and `1,155,584`. Seed `112` is the high line in the `78` image panel.

### Conclusions

A smaller rehearsal does not keep raising the mean. The means are `85.65%`, `89.01%`, `88.24%`, `84.48%`, `81.03%`, `83.62%`, and `83.20%` for `5,000`, `2,500`, `1,250`, `625`, `312`, `156`, and `78` clean images. The `2,500` image mean is the highest, and the three seeds end within `0.66` percentage points. The `312` image mean is the lowest. The `156` image mean is `5.39` percentage points below the `2,500` image mean. The `78` image mean is `5.81` percentage points below it.

The best single seed in this phase is the `1,250` image seed `110`, at `90.12%`. The best single seed in phase 5 is still the `1` epoch seed `111`, at `90.35%`. The lowest seed is the `312` image seed `110`, at `74.00%`.

The phase 1 only-convolution mean is `91.21%`. The phase 4 `1` epoch mean is `88.10%`. The `2,500` image mean is `0.91` percentage points above that phase 4 mean and `2.20` percentage points below the phase 1 mean.

## Phase 7: longer real training

Phase 7 asks whether more SGD in each generation changes the result of the `1,250` image grade. The action list, the Monte Carlo algorithm, the `300` second budget, the `4` simulation epochs, the first `1,250` clean images, the `5,000` validation images, and the extra rollout depth of `0` stay as they are in phase 6. The real training stays `12` generations on the `45,000` augmented images. The only change is the number of epochs inside each generation: `20`, then `30`. Phases 1 through 6 use `10`. The cosine still runs from `0.1` to `0.001`, across `240` epochs for the `20` epoch grade and across `360` epochs for the `30` epoch grade.

| Setting | Value |
| --- | --- |
| Simulation training images | `1,250`, clean, the first `1,250` |
| Simulation validation images | `5,000` |
| Simulation epochs | `4` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Real epochs per generation | `20`, `30` |
| Generations | `12` |
| Output | `exp008_generation_only_phase7/epochs_<N>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at the last epoch of the run. The `10` epoch column is the phase 6 run with the same settings: seeds `110`, `111`, and `112`, `12` generations of `10` epochs, `4` simulation epochs, the first `1,250` clean images, a `300` second budget, and rollout depth `0`. The `20` and `30` epoch columns are the phase 7 runs. Each file stores `simulation_epochs` `4`, `simulation_set_size` `1250`, `simulation_clean_prefix` `1250`, `simulation_time` `300.0`, `simulation_rollout_depth` `0`, and `simulation_augmented` false. The phase 7 files also store `train_epochs` `20` or `30`. Save times, from the filesystem, run from `2026-10-23 20:21` for the `20` epoch seed `110` to `2026-10-24 08:07` for the `30` epoch seed `112`.

| Seed | `10` epochs | `20` epochs | `30` epochs |
| --- | ---: | ---: | ---: |
| `110` | `90.12%` | `86.92%` | `82.79%` |
| `111` | `85.85%` | `82.67%` | `90.91%` |
| `112` | `88.76%` | `84.35%` | `88.74%` |
| Mean | `88.24%` | `84.65%` | `87.48%` |

The `10` epoch mean, `88.24%`, is the highest. The `20` epoch mean is `3.59` percentage points below it. The `30` epoch mean is `0.76` percentage points below it. The `20` epoch last values span `4.25` percentage points. The `30` epoch last values span `8.12` percentage points.

Figure 22 is one row per real training length. The middle column is official test.

![Train, official test, and validation for three training lengths](/assets/experiments/008-phase7-epoch-grid.png)

> [!CAPTION] Figure 22. Three rows are 10, 20, and 30 real epochs in each of 12 generations, on the 1,250-image 4-epoch grade. The 10 epoch row is the phase 6 run. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the 12 generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

Seed `112` at `20` epochs falls from `72.32%` at generation `2` to `60.55%` at generation `4`, then ends at `84.35%`. Seed `111` at `30` epochs is `91.02%` at generation `10` and `90.91%` at generation `11`. That is the highest seed in this comparison. Its training accuracy ends at `97.08%`. Seed `112` at `30` epochs ends with training accuracy `97.43%` and official test `88.74%`.

The `20` epoch seed `111` chooses no convolution, deletes neurons `4` times, and ends at `246,401` parameters. The `30` epoch seed `110` chooses no convolution and ends at `205,342` parameters. The `30` epoch seed `111` chooses `3` sequential convolutions and ends at `728,444` parameters. The `30` epoch seed `112` chooses `2` residual convolutions and `4` neuron adds, and ends at `2,808,473` parameters.

### Conclusions

More epochs in each generation do not raise the mean. The means are `88.24%`, `84.65%`, and `87.48%` for `10`, `20`, and `30` real epochs. The `10` epoch mean is the highest.

The best single seed in this comparison is the `30` epoch seed `111`, at `90.91%`. That is `0.56` percentage points above the phase 5 `1` epoch seed `111`, at `90.35%`. The phase 1 only-convolution mean is `91.21%`. The `30` epoch seed `111` is `0.30` percentage points below that mean.

## Phase 8: how many generations

Phase 8 asks whether a different number of generations changes the result of the `1,250` image grade. Phase 7 found that `10` real epochs per generation had the highest mean, so that length stays. The action list, the Monte Carlo algorithm, the `300` second budget, the `4` simulation epochs, the first `1,250` clean images, the `5,000` validation images, and the extra rollout depth of `0` stay as they are in phase 7. The real training stays on the `45,000` augmented images. The only change is the generation count: `10`, `15`, and `20`. The existing comparison is `12` generations. The last generation trains and does not search, so those counts execute `9`, `11`, `14`, and `19` actions. The cosine still runs from `0.1` to `0.001`, across `100`, `120`, `150`, and `200` epochs.

| Setting | Value |
| --- | --- |
| Simulation training images | `1,250`, clean, the first `1,250` |
| Simulation validation images | `5,000` |
| Simulation epochs | `4` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Real epochs per generation | `10` |
| Generations | `10`, `15`, `20`. The `12` generation column is the phase 6 run |
| Output | `exp008_generation_only_phase8/generations_<N>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at the last epoch of the run. The `12` generation column is the phase 6 run with the same settings. The `10`, `15`, and `20` generation columns are the phase 8 runs. Each file stores `train_epochs` `10`, `simulation_epochs` `4`, `simulation_clean_prefix` `1250`, `simulation_time` `300.0`, and `simulation_rollout_depth` `0`. The phase 8 files also store `train_generations` `10`, `15`, or `20`. Save times, from the filesystem, run from `2026-10-24 10:32` for the `10` generation seed `110` to `2026-10-25 02:45` for the `20` generation seed `112`.

| Seed | `10` generations | `12` generations | `15` generations | `20` generations |
| --- | ---: | ---: | ---: | ---: |
| `110` | `81.78%` | `90.12%` | `90.62%` | `91.50%` |
| `111` | `80.51%` | `85.85%` | `90.51%` | `83.08%` |
| `112` | `79.67%` | `88.76%` | `88.99%` | `90.46%` |
| Mean | `80.65%` | `88.24%` | `90.04%` | `88.35%` |

The `15` generation mean, `90.04%`, is the highest. It is `1.80` percentage points above the `12` generation mean and `1.17` percentage points below the phase 1 only-convolution mean of `91.21%`. The `10` generation mean is `7.59` percentage points below the `12` generation mean. The `20` generation mean is `0.11` percentage points above it. The `15` generation last values span `1.63` percentage points, the smallest span of the four. The `20` generation last values span `8.42` percentage points.

Figure 23 is one row per generation count. The middle column is official test.

![Train, official test, and validation for four generation counts](/assets/experiments/008-phase8-epoch-grid.png)

> [!CAPTION] Figure 23. Four rows are 10, 12, 15, and 20 generations of 10 real epochs, on the 1,250-image 4-epoch grade. The 12 generation row is the phase 6 run. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, and green is seed 112. Official test markers are the generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

The `10` generation grade chooses no convolution on any seed. Seed `112` is `79.86%` at generation `8` and `79.67%` at generation `9`. The `15` generation grade chooses at least one convolution on every seed. Seed `110` chooses `3` sequential convolutions and `6` neuron adds, and ends at `90.62%`. The `20` generation seed `110` chooses `4` sequential convolutions and `2` residual convolutions, and ends at `91.50%`. Seed `111` chooses no convolution and ends at `83.08%`. Seed `112` falls from `80.26%` at generation `3` to `74.77%` at generation `4`, then ends at `90.46%`.

### How did the parameter count move?

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. The last generation does not search, so the last value is the parameter count of the finished network.

| Seed | `10` generations | `12` generations | `15` generations | `20` generations |
| --- | ---: | ---: | ---: | ---: |
| `110` | `531,423` | `1,895,355` | `932,741` | `2,516,716` |
| `111` | `188,121` | `660,871` | `2,252,688` | `458,069` |
| `112` | `162,018` | `545,049` | `1,311,095` | `983,625` |

Figure 24 is one panel per generation count.

![Parameter count by generation count](/assets/experiments/008-phase8-parameters.png)

> [!CAPTION] Figure 24. Parameter count by epoch. Four panels are 10, 12, 15, and 20 generations of 10 real epochs. The 12 generation panel is the phase 6 run. Blue is seed 110, orange is seed 111, and green is seed 112. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count. Every panel uses the same vertical axis.

The `15` generation seed `111` ends at `2,252,688` parameters and `90.51%` official test. The `20` generation seed `111` ends at `458,069` parameters and `83.08%`. The `20` generation seed `110` is the high line in that panel, at `2,516,716` parameters and `91.50%`.

### Conclusions

Ten generations are too few for this grade. The mean is `80.65%`, and none of the three seeds chooses a convolution.

Fifteen generations give the highest mean, `90.04%`. The three seeds end within `1.63` percentage points. Twenty generations do not keep that mean: it falls to `88.35%`, because seed `111` ends at `83.08%` after choosing no convolution.

The best single seed is the `20` generation seed `110`, at `91.50%`. That is `0.29` percentage points above the phase 1 only-convolution mean of `91.21%`. The `15` generation mean remains `1.17` percentage points below that phase 1 mean.

## Phase 10: learning rate and conv placement

Phase 10 asks whether the new learning rate and the new conv placement change the result of the `15` generation grade. Phase 8 found that `15` generations of `10` real epochs had the highest mean, so that length stays. The simulation stays at `4` epochs on the first `1,250` clean images, with a `300` second budget and an extra rollout depth of `0`. Every arm uses the full action list. The starter stays `FourLayerCifarNet` at `128` channels and hidden size `256`. Cutout, label smoothing `0.1`, momentum `0.9`, weight decay `0.0001`, and gradient clip `1.0` stay on. Seeds are `110`, `111`, `112`, `113`, and `114`.

The old learning rate is the composed cosine from `learning_rate_scheduler_factory`. It starts at `0.01`, and an action restarts its warmup. The new learning rate is `CosineDecayScheduler`, from `0.1` to `0.001` across the `150` epochs, and an action does not restart it. The old conv placement sets `seq_conv_insert_before_global_pool` to false, so a new conv sits on the `1×1` pooled map. The new placement sets it to true, so the new conv sits at `8×8`, before the global pool. Search can choose the convolution alone or the convolution followed by BatchNorm and ReLU in either placement.

| Arm | Learning rate | Conv placement | Actions | Seeds |
| --- | --- | --- | --- | --- |
| `base` | old cosine, restarts after an action | after the global pool, no BatchNorm or ReLU | all | `5` |
| `lr` | new cosine, `0.1` to `0.001`, no restart | after the global pool, no BatchNorm or ReLU | all | `5` |
| `conv` | old cosine, restarts after an action | before the global pool, with BatchNorm and ReLU | all | `5` |
| `lr_conv` | new cosine, `0.1` to `0.001`, no restart | before the global pool, with BatchNorm and ReLU | all | `5` |

| Setting | Value |
| --- | --- |
| Generations | `15` |
| Real epochs per generation | `10` |
| Simulation training images | `1,250`, clean, the first `1,250` |
| Simulation epochs | `4` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Actions | all |
| Output | `exp008_generation_only_phase10/<arm>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at the last epoch of the run. Each file stores `train_generations` `15`, `train_epochs` `10`, `simulation_epochs` `4`, `simulation_clean_prefix` `1250`, `simulation_time` `300.0`, and `simulation_rollout_depth` `0`. Save times, from the filesystem, run from `2026-10-26 01:32` for `base` seed `110` to `2026-10-28 16:41` for `lr_conv` seed `114`.

| Seed | `base` | `lr` | `conv` | `lr_conv` |
| --- | ---: | ---: | ---: | ---: |
| `110` | `81.36%` | `82.15%` | `82.14%` | `89.08%` |
| `111` | `81.57%` | `82.44%` | `85.13%` | `89.10%` |
| `112` | `83.27%` | `81.25%` | `83.47%` | `88.93%` |
| `113` | `81.21%` | `87.82%` | `82.28%` | `86.60%` |
| `114` | `83.42%` | `82.48%` | `83.62%` | `86.50%` |
| Mean | `82.17%` | `83.23%` | `83.33%` | `88.04%` |

The `lr_conv` mean, `88.04%`, is the highest. It is `5.87` percentage points above `base`, `4.81` percentage points above `lr`, and `4.71` percentage points above `conv`. `lr` is `1.06` percentage points above `base`. `conv` is `1.16` percentage points above `base`. The `lr_conv` last values span `2.60` percentage points. The `base` span is `2.21` percentage points, and the `conv` span is `2.99`. The `lr` span is `6.57` percentage points, because seed `113` ends at `87.82%` and the other four end between `81.25%` and `82.48%`.

Seeds `110`, `111`, and `112` in `lr_conv` end at `89.08%`, `89.10%`, and `88.93%`. Their mean is `89.04%`. The phase 8 `15` generation mean on those same three seeds is `90.04%`. The phase 10 mean is `1.00` percentage point below it. Seeds `113` and `114` end at `86.60%` and `86.50%`, so the five-seed mean is `88.04%`.

Figure 25 is one row per arm. The middle column is official test.

![Train, official test, and validation for the four phase 10 arms](/assets/experiments/008-phase10-epoch-grid.png)

> [!CAPTION] Figure 25. Four rows are the arms base, lr, conv, and lr_conv. Three columns are training accuracy, official test accuracy, and validation accuracy, in percent. Blue is seed 110, orange is seed 111, green is seed 112, purple is seed 113, and red is seed 114. Official test markers are the generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

`lr_conv` is the only row whose official-test lines finish near `90%`. `base`, `lr`, and `conv` stay in a band around `80%` to `85%`, except `lr` seed `113`. On `lr_conv`, the mean training accuracy ends at `91.78%` and the mean official test ends at `88.04%`. On `conv`, the mean training accuracy ends at `81.65%`, below the mean official test of `83.33%`.

| Seed | `base` train | `lr` train | `conv` train | `lr_conv` train | `base` validation | `lr` validation | `conv` validation | `lr_conv` validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `110` | `77.40%` | `84.69%` | `80.61%` | `91.57%` | `80.66%` | `81.90%` | `81.94%` | `89.00%` |
| `111` | `82.85%` | `84.24%` | `81.80%` | `93.33%` | `81.68%` | `82.30%` | `84.94%` | `88.82%` |
| `112` | `84.91%` | `82.52%` | `84.08%` | `94.22%` | `83.12%` | `81.00%` | `83.08%` | `89.10%` |
| `113` | `81.95%` | `94.88%` | `79.81%` | `88.58%` | `81.36%` | `87.52%` | `82.06%` | `86.60%` |
| `114` | `85.07%` | `84.56%` | `81.97%` | `91.22%` | `82.84%` | `82.22%` | `83.44%` | `86.34%` |
| Mean | `82.44%` | `86.18%` | `81.65%` | `91.78%` | `81.93%` | `82.99%` | `83.09%` | `87.97%` |

Figure 26 is the five-seed mean of those three curves.

![Five-seed mean train, official test, and validation](/assets/experiments/008-phase10-mean-curves.png)

> [!CAPTION] Figure 26. Five-seed mean accuracy by epoch. Three panels are training accuracy, official test accuracy, and validation accuracy, in percent. Gray is base, blue is lr, orange is conv, and green is lr_conv. Official test markers are the generation-end measurements. The dashed line is 90%. Every panel uses the same vertical axis, from 0% to 100%.

The green line separates from the other three during training and stays above them on validation. On official test, the green mean stays above the other three from the first marker. The orange mean starts lower, then the gray, blue, and orange means finish close together.

Figure 27 is the last official test of each seed, with the five-seed mean drawn over them.

![Last official test by arm](/assets/experiments/008-phase10-final-test.png)

> [!CAPTION] Figure 27. Official test accuracy at epoch 150, in percent. The horizontal axis is the arm. Blue is seed 110, orange is seed 111, green is seed 112, purple is seed 113, and red is seed 114. Those seed lines are drawn lighter than the mean. The black line is the five-seed mean. The dotted line is 90%.

The black line steps up only at `lr_conv`. Purple is the high seed on `lr` and one of the two low seeds on `lr_conv`.

### How did the learning rate move?

`lr` is the learning rate stored on each training epoch. Within one arm the five seeds stored the same list. `base` and `conv` use `ComposedLearningRateScheduler`. The saved rate starts at `0.01`. After an action it falls to `0.001` and climbs back toward `0.01`. An action runs after every generation except the last, so that drop repeats through the run. `lr` and `lr_conv` use `CosineDecayScheduler`. The saved rate starts at `0.1` and falls to `0.001` across the `150` epochs. An action does not restart it. The last value of every arm is `0.001`.

Figure 28 is one panel per arm. The vertical axis is shared, so the old rate occupies the bottom tenth of the panel.

![Learning rate by arm](/assets/experiments/008-phase10-learning-rate.png)

> [!CAPTION] Figure 28. Learning rate by epoch. Four panels are base, lr, conv, and lr_conv. Each panel has one line because the five seeds stored the same rate. The vertical axis is shared. The rate has no percent unit.

The two old panels show the repeated drop and climb. The two new panels show one smooth fall from `0.1` to `0.001`.

### How did the parameter count move?

`param_count` is recorded after each generation's SGD and before that generation's action, then copied onto every epoch of the generation. The last generation does not search, so the last value is the parameter count of the finished network. Every seed executes `14` actions.

| Seed | `base` | `lr` | `conv` | `lr_conv` |
| --- | ---: | ---: | ---: | ---: |
| `110` | `1,943,427` | `379,891` | `2,819,658` | `375,602` |
| `111` | `720,960` | `576,079` | `627,554` | `434,117` |
| `112` | `1,630,592` | `167,048` | `1,413,568` | `1,336,601` |
| `113` | `892,928` | `4,306,551` | `1,957,822` | `2,057,949` |
| `114` | `3,229,493` | `820,776` | `1,721,748` | `2,007,296` |

Figure 29 is one panel per arm.

![Parameter count by arm](/assets/experiments/008-phase10-parameters.png)

> [!CAPTION] Figure 29. Parameter count by epoch. Four panels are base, lr, conv, and lr_conv. Blue is seed 110, orange is seed 111, green is seed 112, purple is seed 113, and red is seed 114. The count is flat inside a generation and changes at the first epoch after an action that changes the parameter count. Every panel uses the same vertical axis.

On `lr`, four seeds end below `1,000,000` parameters and choose no convolution. Seed `113` chooses one residual convolution, ends at `4,306,551` parameters, and is the `87.82%` seed. On `lr_conv`, seeds `110` and `111` end at `375,602` and `434,117` parameters and at `89.08%` and `89.10%`. Seeds `113` and `114` end near `2,000,000` parameters and at `86.60%` and `86.50%`.

`lr` seed `114` falls from `75.02%` at generation `6` to `69.52%` at generation `7`, then ends at `82.48%`. `lr_conv` seed `113` falls from `69.44%` at generation `5` to `64.09%` at generation `6`, then ends at `86.60%`. `lr_conv` seed `114` falls from `68.81%` at generation `1` to `62.95%` at generation `2`, and from `69.40%` at generation `3` to `63.06%` at generation `4`, then ends at `86.50%`.

### Conclusions

The two changes together give the highest mean. `lr_conv` ends at `88.04%`. Either change alone stays near the arm with both changes off: `lr` ends at `83.23%` and `conv` ends at `83.33%`, against `82.17%` for `base`.

The new learning rate alone does not raise the four seeds that choose no convolution. Those four end at `82.15%`, `82.44%`, `81.25%`, and `82.48%`. Their mean is `82.08%`, `0.09` percentage points below the `base` mean.

The best single seed is `lr_conv` seed `111`, at `89.10%`. That is `1.52` percentage points below the phase 8 `15` generation seed `110`, at `90.62%`. It is `2.11` percentage points below the phase 1 only-convolution mean of `91.21%`.

## Phase 11: one change at a time

Phase 11 asks which one of the remaining changes is carrying the phase 10 `base` result. Phase 10 turned the learning rate and the conv placement off together and the mean stayed at `82.17%`. Those two stay off. Every arm copies `base` and changes one setting. The control is not rerun. Seeds `110` and `111` of phase 10 `base` ended at `81.36%` and `81.57%`.

The copied recipe is the old composed cosine from `0.01`, with a warmup restart after an action, and `seq_conv_insert_before_global_pool` false, so a new conv sits after the global pool. Search can still choose the convolution alone or the convolution followed by BatchNorm and ReLU. All actions stay on, with `15` generations of `10` epochs, simulation `4` epochs on the first `1,250` clean images, `300` seconds, and rollout depth `0`. `FourLayerCifarNet` at `128` channels and hidden size `256`, cutout, label smoothing `0.1`, gradient clip `1.0`, weight decay `0.0001`, and search after every generation except the last stay on unless the arm names that setting.

| Arm | The one change | Seeds |
| --- | --- | ---: |
| `minimal` | `MinimalCifarNet`, `32` channels, hidden size `64`, no BatchNorm | `2` |
| `slope` | search only when the training slope is under `3` degrees | `2` |
| `no_cutout` | cutout off | `2` |
| `no_smooth` | label smoothing `0` | `2` |
| `no_clip` | gradient clip off | `2` |
| `old_decay` | weight decay `0.0005` | `2` |
| `short` | `8` generations of `10` epochs | `2` |

| Setting | Value |
| --- | --- |
| Control | phase 10 `base`, seeds `110` and `111` |
| Simulation training images | `1,250`, clean, the first `1,250` |
| Simulation epochs | `4` |
| Simulation time | `300` seconds |
| Extra actions inside the score | `0` |
| Actions | all |
| Output | `exp008_generation_only_phase11/<arm>/seed_<seed>` |

Official test accuracy is the fraction correct on the `10,000` image test set, at the last epoch of the run. The new cells stay empty until `history.pt` exists.

| Arm | Seed `110` | Seed `111` |
| --- | ---: | ---: |
| `base` | `81.36%` | `81.57%` |
| `minimal` |  |  |
| `slope` |  |  |
| `no_cutout` |  |  |
| `no_smooth` |  |  |
| `no_clip` |  |  |
| `old_decay` |  |  |
| `short` |  |  |

No charts are drawn until those files exist. The training run is not started.
