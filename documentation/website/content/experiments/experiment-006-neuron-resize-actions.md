# Experiment 006: Neuron-resize action ratio pairs

We keep the Experiment 005 / train-ci package (`sequential_halving_beam` + `composed_exponential` + `big` starter). We change only which AddNeurons / DelNeurons ratio pair is enabled.

The goal is to learn whether the six neuron-resize flags in `RunningConfig` are all useful on a short MNIST probe, or whether some ratio pairs add nothing and should stay off.

Script: `experiments/train_mnist_exp006_neuron_resize_actions.py`

Charts: `documentation/website/scripts/generate_experiment_006_charts.py`

Folder: `experiments/output/train_mnist/runs/exp006_neuron_resize_actions`

Snapshot: `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`

This page is a report template. Fill tables and conclusions after the grid finishes. Charts appear once `generate_experiment_006_charts.py` has boards or a snapshot.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Neuron-resize group | `none`, `add11_del01`, `add15_del05`, `add20_del09` | Compare no width change vs mild / medium / aggressive ratio pairs |
| Seed | `100`, `101`, `102` | Three matched seeds per group |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `12` | `4` groups × `3` seeds |
| Simulation algorithm | `sequential_halving_beam` | Best keep-set method from Experiment 005 |
| LR package | `composed_exponential` × logistic recovery | Best package from Experiment 004 |
| Standard cell `lr_alpha` | `0.01` | Target / peak learning rate |
| Accuracy metric | `val_acc` | Simulation grading |
| Slope threshold | `3°` | `SlopeEstimationSimulationScheduler` gate |
| Generations | `8` | Short probe (below Exp 005 `10`) |
| Epochs per generation | `8` | Short probe (below Exp 005 `10`) |
| Total training epochs | `64` | `8 × 8` |
| Simulation time | `120 s` | Same budget as Exp 005 |
| Starter | `big` (`BigAvgPoolMnistNet`) | Same as Exp 004 / train-ci |
| Layer add / delete / dropout | on | Only neuron-resize flags vary |

Group meanings:

| Group ID | Enabled flags | Ratios |
| --- | --- | --- |
| `none` | none | control, Exp 001–005 style |
| `add11_del01` | `ADD_NEURONS_11`, `DEL_NEURONS_01` | grow ×1.1, shrink ×0.1 |
| `add15_del05` | `ADD_NEURONS_15`, `DEL_NEURONS_05` | grow ×1.5, shrink ×0.5 |
| `add20_del09` | `ADD_NEURONS_20`, `DEL_NEURONS_09` | grow ×2.0, shrink ×0.9 |

Run path:

```text
exp006_neuron_resize_actions/<group_id>/<hp_folder>/seed_<seed>/
```

## Research questions

Main question: which neuron-resize ratio pair improves short MNIST GrowingNN runs enough to keep enabled by default?

Supporting checks:

1. Does any enabled pair beat the `none` control on final validation accuracy?
2. Do mild / medium / aggressive pairs get selected by search, or do they sit unused?
3. Does enabling a pair grow or shrink parameter count in a useful way?
4. Should default config keep all three pairs, keep one pair, or keep none?

## Result timeline

Progress: `0` / `12` completed (`0.0%`). Re-run this section after the grid finishes.

| Group | Seeds done | Notes |
| --- | ---: | --- |
| `none` | `0` / `3` | control |
| `add11_del01` | `0` / `3` | mild pair |
| `add15_del05` | `0` / `3` | medium pair |
| `add20_del09` | `0` / `3` | aggressive pair |

## Why this experiment

Experiments 001–005 kept AddNeurons / DelNeurons off while layer resize was unstable. After the layer-resize fix, `RunningConfig` enables all six neuron-resize flags by default. Exp 001–005 still force them off locally so old grids stay comparable.

This short probe asks whether those six flags deserve to stay on for new runs. We test them as three paired ratio groups against a no-resize control, instead of enabling all six at once.

## Measurements and charts

Generate charts after runs exist:

```text
python documentation/website/scripts/generate_experiment_006_charts.py
```

### Final accuracy by group

![Final accuracy by neuron-resize group](/assets/experiments/006-final-accuracy-by-group.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy by neuron-resize group. Gray markers are individual seeds.

### Parameter growth by group

![Parameter growth by neuron-resize group](/assets/experiments/006-param-growth-by-group.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by group. Gray markers are individual final counts.

### Action mix by group

![Executed action mix by neuron-resize group](/assets/experiments/006-action-composition-by-group.png)

> [!CAPTION] Figure 3. Mean executed action counts by short label and group. Neuron-resize labels should appear only when that group enables them.

### Training curves by group

![Training accuracy curves by neuron-resize group](/assets/experiments/006-training-curves.png)

> [!CAPTION] Figure 4. Training accuracy over epochs for every completed seed, colored by group.

## Grouped final results

Fill after completion.

| Group | Mean train (%) | Mean val (%) | Mean final params | Neuron-resize actions used |
| --- | ---: | ---: | ---: | ---: |
| `none` |  |  |  |  |
| `add11_del01` |  |  |  |  |
| `add15_del05` |  |  |  |  |
| `add20_del09` |  |  |  |  |

## Limitations and seed effects

- Short run (`64` epochs) can understate late width changes.
- Three seeds are enough for a first ranking, not for a hard reject of a close second place.
- MNIST `big` may need fewer growth steps than a harder starter, so a useful pair here should still be re-checked on a harder task.

## Conclusions

To fill after the grid:

1. State which group beats `none` on mean validation accuracy.
2. State whether unused ratio pairs should be turned off in default config.
3. State the recommended default: keep all three pairs, keep one pair, or keep none.

## Next experiments

1. If one pair wins clearly, re-test it with Exp 005 length (`10×10`) and the medium starter.
2. If all three pairs help, keep the six default flags on and move to a harder dataset.
3. If none beats the control, turn neuron-resize defaults back off and keep only layer add/delete.
