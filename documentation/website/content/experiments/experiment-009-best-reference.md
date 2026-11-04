# Experiment 009: the 91% CIFAR-10 recipe with every action left on

This experiment asks what the Experiment 008 reference does when every architecture action stays enabled, and what that same loop does on MNIST, CIFAR-100, and Tiny ImageNet.

On CIFAR-10 it does not repeat the 91% result. Seeds `1` and `2` finish at `82.06%` and `80.70%` official test. The same loop reaches `98.51%` and `98.21%` on MNIST, `50.22%` and `58.87%` on CIFAR-100, and `44.46%` and `47.14%` on Tiny ImageNet.

A later run trains for 20 generations. MNIST keeps the one-convolution starter. CIFAR-10 uses the 187,008-parameter starter. The finished files from that folder are in Twenty generations with the current starters.

Script: `experiments/train_cifar10_exp009_best_reference.py`

Run output: `experiments/output/train_cifar10/runs/exp009_best_reference`

Charts: `documentation/website/scripts/generate_experiment_009_charts.py`

Snapshot: `documentation/website/data/experiments/experiment-009-best-reference.json`

The 20-generation run has its own snapshot: `documentation/website/data/experiments/experiment-009-small-20.json`

The raw run files are gitignored. Commit the page, the charts, and the snapshot before deleting `experiments/output/`.

## Fixed training loop

`train_seed` builds one `RunningConfig` for every dataset. `train_generations` in `growingnn/training/trainer.py` runs it.

- Generations × epochs: `12` × `10`. The last generation does not search.
- Learning rate: `CosineDecayScheduler(total_epochs=120)` from `0.1` to `0.001`. `structure_changed` does not restart the cosine.
- Optimizer: SGD, momentum `0.9`, weight decay `0.0001`, `max_grad_norm` `1.0`.
- Loss: `CrossEntropyLoss(label_smoothing=0.1)`. Accuracy is still argmax.
- Search: `AlwaysSimulationScheduler(8.0, 1)`, Monte Carlo, `simulation_set_size` `128`, `SimulationScore(weight_acc=1.0, weight_countW=0.0)`.
- Cutout: `with_cutout` wraps the training loader.
- `seq_conv_insert_before_global_pool` is `True`.
- Every `RunningConfig` action flag stays at its default `True`. That includes sequential and residual convolutions, linear layers, dropout, neuron add and delete, and layer delete.
- Seeds are `SEEDS = (1, 2)`.

Official test is measured after each generation's SGD and before the action. Search does not read the official test loader.

## What each dataset opens

`get_datasets` calls one `DataFactory` method. `initial_graph` calls the matching `InitModelFactory` method.

| Dataset | Starter | Start parameters | Train / validation | Official test |
| --- | --- | ---: | --- | --- |
| MNIST | `InitModelFactory.MnistStarter` | 12,256 | 50,000 / 10,000 | 10,000 |
| CIFAR-10 | `FourLayerCifarNet`, 128 channels, hidden 256 | 187,008 | 45,000 / 5,000 | 10,000 |
| CIFAR-100 | `InitModelFactory.Cifar100Starter` | 123,744 | 45,000 / 5,000 | 10,000 |
| Tiny ImageNet | `InitModelFactory.TinyImageNetStarter` | 160,064 | 90,000 / 10,000 | 10,000 |

CIFAR-10 uses `Cifar10Data`. The others use `DataFactory.SplitData` with `random_split` seed `0`. Batch size is `128`. Tiny ImageNet is read from `experiments/data/tiny-imagenet-200`. ImageNet-100 and ImageNet are in `DATASETS` but have no `history.pt` yet.

## Script and timeline

Dates below are filesystem modification times of `history.pt`, not a training log.

- CIFAR-10 seed `1`: `2026-11-01 22:20:16`. Seed `2`: `22:40:01`.
- CIFAR-100 seed `1`: `2026-11-01 23:38:36`. Seed `2`: `2026-11-02 00:55:42`.
- MNIST seed `1`: `2026-11-02 01:18:04`. Seed `2`: `01:39:52`.
- Tiny ImageNet seed `1`: `2026-11-02 11:14:00`. Seed `2`: `15:26:14`.

## What each training curve does

Training accuracy is measured every epoch on the augmented loader. Validation is the clean held-out slice from the training set. Vertical lines are generation boundaries, where one action may be committed. Official test is not on this chart.

![Training and validation by epoch](/assets/experiments/009-train-histories.png)

> [!CAPTION] Figure 1. Accuracy in percent for all 120 epochs. Panels run from the smallest dataset to the largest: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet. Solid lines are training. Dashed lines are validation. Blue is seed 1. Orange is seed 2. Pale vertical lines mark epochs 10, 20, ..., 110.

MNIST training stays in a narrow band. Seed 1 moves from `86.99%` to `92.94%`. Seed 2 dips from `88.96%` at epoch 30 to `87.42%` at epoch 40, then ends at `92.15%`. Validation sits near `98%` for the whole run, about six points above training.

CIFAR-10 seed 1 climbs through the middle generations, reaches `80.14%` training accuracy at epoch 110, and falls to `77.02%` during the last ten epochs. Seed 2 does not show that late drop. It moves from `60.37%` at epoch 10 to `81.29%` at epoch 120, with a flat stretch near `65%` between epochs 30 and 40. Validation stays beside training for both seeds.

CIFAR-100 seed 1 rises in small steps and ends at `46.23%` training accuracy. Seed 2 falls from `29.14%` at epoch 10 to `25.75%` at epoch 20, falls again to `25.76%` at epoch 40, and only then climbs, passing seed 1 after epoch 80 and finishing at `55.56%`.

Tiny ImageNet seed 2 pulls ahead in the second generation and stays above seed 1. Its training accuracy goes from `21.00%` at epoch 10 to `50.57%` at epoch 120. Seed 1 stays near `27%` from epoch 40 to epoch 60, then steps up and ends at `45.66%`. Validation follows training and finishes just below it.

## Did the reference still pass 90%?

Official test accuracy is the fraction correct on the dataset's held-out test images. The value below is the measurement after generation 12.

![Final official test by dataset](/assets/experiments/009-final-official-test.png)

> [!CAPTION] Figure 2. Last official test accuracy in percent. Bars run from the smallest dataset to the largest: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet. Blue is seed 1. Orange is seed 2. Each bar is one finished run.

| Dataset | Seed 1 test | Seed 2 test | Mean test | Seed 1 train | Seed 2 train | Seed 1 validation | Seed 2 validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | 98.51% | 98.21% | 98.36% | 92.94% | 92.15% | 98.37% | 98.09% |
| CIFAR-10 | 82.06% | 80.70% | 81.38% | 77.02% | 81.29% | 82.04% | 80.24% |
| CIFAR-100 | 50.22% | 58.87% | 54.55% | 46.23% | 55.56% | 49.56% | 58.50% |
| Tiny ImageNet | 44.46% | 47.14% | 45.80% | 45.66% | 50.57% | 44.55% | 46.89% |

MNIST is already above 96% after the first generation. No CIFAR-10 seed reaches 90%. The two CIFAR-10 seeds agree within `1.36` percentage points. CIFAR-100 does not: seed 2 is `8.65` percentage points above seed 1. Tiny ImageNet rises on both seeds and finishes in the mid-40s.

Training accuracy is measured on augmented images with label smoothing. On MNIST it stays near 92% while the clean official test is near 98%. On CIFAR-10 seed 1 the last training accuracy, `77.02%`, is below the official test.

## Where the CIFAR-10 curves separate from the 91% runs

The chart is the official test after each generation. The dashed line on the CIFAR-10 panel is the Experiment 008 sequential-convolution mean, `91.21%`. It is not a run from this script.

![Official test after each generation](/assets/experiments/009-test-by-generation.png)

> [!CAPTION] Figure 3. Official test accuracy in percent at epochs 10, 20, ..., 120. Panels run from the smallest dataset to the largest: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet. Blue is seed 1. Orange is seed 2. The dashed line is 91.21% and appears only on CIFAR-10.

CIFAR-10 seed 1 rises from `60.84%` at epoch 10 to `82.06%` at epoch 120. The rise slows after epoch 70, when the test is `74.90%`. Seed 2 reaches `68.41%` at epoch 20, then drops to `65.02%` by epoch 40, and ends at `80.70%` after a `0.32` point drop on the last generation.

The Experiment 008 sequential-convolution runs, seeds `110`, `111`, and `112`, were already near `88%` by epoch 70 and finished at `91.11%`, `91.17%`, and `91.36%`. Those runs grew from `187,008` to `1,813,248` parameters. This experiment keeps the same `FourLayerCifarNet`, the same 120-epoch cosine, and the same short search, and turns the other actions back on.

The saved action names show what the short search committed:

- CIFAR-10 seed 1: `AddNeurons`, `DelNeurons`, `AddSeqDropoutLayer`, `DelNeurons`, `AddSeqConvLayer`, `AddNeurons`, `AddSeqConvLayer`, then three `AddResConvLayer`, then `DelNeurons`. End size `721,316`.
- CIFAR-10 seed 2: `AddSeqDropoutLayer`, `DelNeurons`, four sequential or residual linear layers, and three `AddNeurons`. No convolution was added. End size `380,364`.

Experiment 008 already recorded this gap with the same trainer. Sequential convolution averaged `91.21%`. All-action search averaged `85.68%`, with seeds at `85.70%`, `90.40%`, and `80.94%`. The two seeds here, `82.06%` and `80.70%`, sit at the low end of that earlier range. The action list is an observation of what search selected. It is not, by itself, proof that one named action caused the whole gap.

## How large the grown graphs became

Parameter count is the number of trainable parameters at the end of each generation.

![Parameters at each generation](/assets/experiments/009-parameters.png)

> [!CAPTION] Figure 4. Trainable parameters, in millions, at epochs 10, 20, ..., 120. Panels run from the smallest dataset to the largest: MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet. Blue is seed 1. Orange is seed 2.

| Dataset | Seed | Start | End | Reference model | Reference size | Times smaller |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| MNIST | 1 | 12,256 | 51,918 | LeNet-5 | 62 thousand | 1.2 |
| MNIST | 2 | 12,256 | 70,208 | LeNet-5 | 62 thousand | 0.9 |
| CIFAR-10 | 1 | 187,008 | 721,316 | ResNet-18 | 11.2 million | 15.5 |
| CIFAR-10 | 2 | 187,008 | 380,364 | ResNet-18 | 11.2 million | 29.4 |
| CIFAR-100 | 1 | 123,744 | 229,494 | ResNet-18 | 11.2 million | 48.8 |
| CIFAR-100 | 2 | 123,744 | 3,354,912 | ResNet-18 | 11.2 million | 3.3 |
| Tiny ImageNet | 1 | 160,064 | 932,032 | ResNet-18 | 11.3 million | 12.1 |
| Tiny ImageNet | 2 | 160,064 | 850,368 | ResNet-18 | 11.3 million | 13.3 |

The reference sizes are the usual published counts for a model that learns that dataset from scratch. They were not trained in this repository. A usual ResNet-18 finish is about `95%` on CIFAR-10, about `76%` on CIFAR-100, and about `60%` on Tiny ImageNet. LeNet-5 is about `99.2%` on MNIST.

CIFAR-100 seed 2 is the only run that grows into the millions. It also has the higher CIFAR-100 test, `58.87%` against `50.22%`. MNIST seed 2 ends slightly larger than LeNet-5. Its test, `98.21%`, is about one point below that baseline.

## Seed and dataset limits

- Two seeds are finished for four datasets. ImageNet-100 and ImageNet have no `history.pt`.
- CIFAR-10 seed 2 never adds a convolution. Its curve cannot be read as the sequential-convolution reference.
- CIFAR-100 seed 2 ends `14.6` times larger than seed 1. The mean `54.55%` hides that split.
- Tiny ImageNet seed 2 finishes `15:26:14`, about four hours after seed 1. Both stay under one million parameters.
- The `91.21%` line is a published mean from another action list. This script did not rerun that list.

## Conclusion

The 120-epoch cosine, the 128-channel CIFAR-10 starter, cutout, and the 8-second search are in place. With every action enabled, CIFAR-10 finishes at `81.38%` mean official test, `9.83` percentage points below the sequential-convolution mean of `91.21%`. The grown CIFAR-10 graphs stay at `721,316` and `380,364` parameters, not `1,813,248`.

MNIST reaches `98.36%` at LeNet scale. CIFAR-100 reaches `54.55%` and Tiny ImageNet reaches `45.80%`, both below a usual ResNet-18 trained on the same data, and both with much smaller graphs except CIFAR-100 seed 2.

## Twenty generations with the current starters

This folder is the 20-generation rerun. `GENERATIONS` is `20` and `EPOCHS` is `10`. `CosineDecayScheduler(total_epochs=200)` runs from `0.1` to `0.001`, and `structure_changed` does not restart it. Cutout, label smoothing `0.1`, momentum `0.9`, weight decay `0.0001`, gradient clip `1.0`, and the 8-second search are the same as the first run. Every action flag stays `True`. Seeds stay `1` and `2`.

`initial_graph` returns one `InitModelFactory` class per dataset. MNIST is one convolution and one linear layer. CIFAR-10 is two 128-channel convolutions, batch norm, a hidden layer of `256`, and a 10-way linear layer. CIFAR-100 and Tiny ImageNet keep those convolutions. Their hidden layers are `512` and `768`.

| Dataset | Starter | Hidden layer | Start parameters | Train / validation | Official test |
| --- | --- | ---: | ---: | --- | --- |
| MNIST | `MnistStarter` | none | 152 | 50,000 / 10,000 | 10,000 |
| CIFAR-10 | `Cifar10Starter` | 256 | 187,008 | 45,000 / 5,000 | 10,000 |
| CIFAR-100 | `Cifar100Starter` | 512 | 268,672 | 45,000 / 5,000 | 10,000 |
| Tiny ImageNet | `TinyImageNetStarter` | 768 | 404,096 | 90,000 / 10,000 | 10,000 |

Run output: `experiments/output/train_cifar10/runs/exp009_small_20`

Dates below are filesystem modification times of `history.pt`. A seed with no file is not in the charts.

- MNIST seed `1`: `2026-11-02 22:29:41`. Seed `2`: `23:56:18`. Both files still start at `152` parameters. The rerun did not replace them.
- CIFAR-10 seed `2`: `2026-11-03 11:03:23`. It starts at `187,008` parameters. Seed `1` has a folder from `08:19:05` and no `history.pt`.
- CIFAR-100 seed `1` has a folder from `11:03:23` and no `history.pt`. Seed `2` has no folder.
- Tiny ImageNet has no folder.

### What the 200-epoch curves do

Training accuracy is measured every epoch on the augmented loader. Validation is the clean held-out slice. Vertical lines are generation boundaries. Official test is not on this chart.

![Training and validation for the finished 20-generation runs](/assets/experiments/009-small20-train-histories.png)

> [!CAPTION] Figure 5. Accuracy in percent for all 200 epochs. Panels are the finished datasets, MNIST then CIFAR-10. Solid lines are training. Dashed lines are validation. Blue is seed 1. Orange is seed 2. CIFAR-10 has seed 2 only. Pale vertical lines mark epochs 10, 20, ..., 190.

MNIST still starts well below its final accuracy. Seed 1 training moves from `38.49%` at epoch 10 to `87.83%` at epoch 200. Validation ends at `96.30%`. Training drops at two generation boundaries, from `73.86%` at epoch 30 to `55.42%` at epoch 31, and from `74.17%` at epoch 40 to `59.73%` at epoch 41. Both drops recover before the next boundary. Seed 2 training reaches `91.22%` at epoch 140, falls to `68.71%` at epoch 150, and ends at `71.53%`. Validation does not follow that drop. It is `96.10%` at epoch 140 and `97.83%` at epoch 200. The action committed at the epoch-140 boundary is `AddSeqDropoutLayer`.

CIFAR-10 seed 2 training moves from `60.50%` at epoch 10 to `90.70%` at epoch 200. Validation stays beside it through the middle of training and finishes at `85.82%`, `4.88` percentage points below training. The curve has no late collapse.

![CIFAR-10 seed 2 accuracy](/assets/experiments/009-small20-cifar10-seed2.png)

> [!CAPTION] Figure 6. CIFAR-10 seed 2 accuracy in percent. The orange solid line is training. The orange dashed line is validation. Blue markers are the official test after each generation. The gray dashed line is 91.21%. Pale vertical lines mark epochs 10, 20, ..., 190.

### Where official test finishes

Official test accuracy is the fraction correct on the held-out test images after generation 20.

![Final official test for the finished 20-generation runs](/assets/experiments/009-small20-final-official-test.png)

> [!CAPTION] Figure 7. Last official test accuracy in percent for the finished histories. Blue is seed 1. Orange is seed 2. CIFAR-10 has seed 2 only.

| Dataset | Seed 1 test | Seed 2 test | Seed 1 train | Seed 2 train | Seed 1 validation | Seed 2 validation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | 96.77% | 98.09% | 87.83% | 71.53% | 96.30% | 97.83% |
| CIFAR-10 | no file | 86.04% | no file | 90.70% | no file | 85.82% |

MNIST seed 1 is `44.37%` after the first generation and ends at `96.77%`. The two MNIST tests differ by `1.32` percentage points. CIFAR-10 seed 2 ends at `86.04%`. That is `3.98` percentage points above the first run's CIFAR-10 seed 1 and `5.34` percentage points above its seed 2. It is `5.17` percentage points below the Experiment 008 sequential-convolution mean of `91.21%`.

On MNIST seed 2, training accuracy ends `26.56` percentage points below the official test. CIFAR-10 seed 2 training ends `4.66` percentage points above its official test.

### How test moves across 20 generations

The chart is the official test after each generation. The dashed line on CIFAR-10 is still the Experiment 008 sequential-convolution mean, `91.21%`.

![Official test for the finished 20-generation runs](/assets/experiments/009-small20-test-by-generation.png)

> [!CAPTION] Figure 8. Official test accuracy in percent at epochs 10, 20, ..., 200. Panels are MNIST and CIFAR-10. Blue is seed 1. Orange is seed 2. CIFAR-10 has seed 2 only. The dashed line is 91.21% and appears only on CIFAR-10.

MNIST seed 1 rises from `44.37%` at epoch 10 to `96.77%` at epoch 200, with a dip from `87.88%` at epoch 40 to `83.56%` at epoch 50. Seed 2 is already `95.43%` at epoch 50 and ends at `98.09%`. The last ten generations add less than one point.

CIFAR-10 seed 2 starts at `62.48%` after the first generation. It dips from `67.75%` at epoch 40 to `63.45%` at epoch 50, then rises on every later generation and ends at `86.04%`. The first convolution, `AddSeqConvLayer`, is committed at the epoch-70 boundary, when the test is `71.82%`. Three of the 19 committed actions are convolutions. The other actions are neuron edits, dropout, and linear layers.

### How large the finished graphs grew

Parameter count is the number of trainable parameters at the end of each generation.

![Parameters for the finished 20-generation runs](/assets/experiments/009-small20-parameters.png)

> [!CAPTION] Figure 9. Trainable parameters, in thousands, at epochs 10, 20, ..., 200. Panels are MNIST and CIFAR-10. Blue is seed 1. Orange is seed 2. CIFAR-10 has seed 2 only.

| Dataset | Seed | Start | End | Reference model | Reference size | Times smaller |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| MNIST | 1 | 152 | 6,088 | LeNet-5 | 62 thousand | 10.2 |
| MNIST | 2 | 152 | 9,016 | LeNet-5 | 62 thousand | 6.9 |
| CIFAR-10 | 2 | 187,008 | 1,532,336 | ResNet-18 | 11.2 million | 7.3 |

The reference sizes are the same published counts as in the first run. They were not trained here.

CIFAR-10 seed 2 grows from `187,008` to `1,532,336` parameters. The first run's CIFAR-10 graphs ended at `721,316` and `380,364`. The Experiment 008 sequential-convolution graphs ended at `1,813,248`. This seed is in that range and still `5.17` percentage points below `91.21%`.

MNIST seed 2 deletes one layer and then keeps growing. It falls from `5,512` parameters at epoch 110 to `4,928` at epoch 120, then ends at `9,016`. Its test stays above `97%` across that deletion.

### What this folder does not show

- CIFAR-10 seed 1, both CIFAR-100 seeds, and both Tiny ImageNet seeds have no `history.pt`.
- The MNIST files are the 2 November run. They are not a second draw of the same starter.
- CIFAR-10 seed 2 is one all-action path. It does not replace the three-seed sequential-convolution mean.
- The action committed at epoch 70 is the first convolution. That timing is not proof that the convolution caused the later rise.

CIFAR-10 seed 2, started from the `187,008`-parameter network and trained for 20 generations with every action enabled, finishes at `86.04%` official test and `1,532,336` parameters. MNIST, from `152` parameters, finishes at `96.77%` and `98.09%`.

## Next experiment

Read CIFAR-10 seed 1, both CIFAR-100 seeds, and both Tiny ImageNet seeds when each `history.pt` exists. Those files are the rest of this rerun. The hidden sizes and the action list stay as they are until those histories are in the folder.
