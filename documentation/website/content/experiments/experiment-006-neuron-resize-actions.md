# Experiment 006: Neuron-resize action ratio pairs

We keep the Experiment 005 / train-ci package (`sequential_halving_beam` + `composed_exponential` + `big` starter). We change only which AddNeurons / DelNeurons ratio pair is enabled.

The goal is to learn whether the six neuron-resize flags in `RunningConfig` are useful on a short MNIST probe, or whether search ignores them and they should stay off.

Script: `experiments/train_mnist_exp006_neuron_resize_actions.py`

Charts: `documentation/website/scripts/generate_experiment_006_charts.py`

Folder: `experiments/output/train_mnist/runs/exp006_neuron_resize_actions`

Snapshot: `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`

Simulation-candidate analysis: `documentation/website/data/experiments/experiment-006-simulation-action-analysis.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`11` / `12` cells, `91.7%`). Simulation tables use `board/simulations/simulation_gen_*.json` from those completed runs (`56` simulation calls).

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Neuron-resize group | `none`, `add11_del01`, `add15_del05`, `add20_del09` | Compare no width change vs mild / medium / aggressive ratio pairs |
| Seed | `100`, `101`, `102` | Three matched seeds per group |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `12` | `4` groups × `3` seeds |
| Completed cells in this refresh | `11` | Missing only stale `add15_del05` seed `101` |
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
| Start params | `420` | Shared starter size |
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

Progress in this refresh: `11` / `12` completed (`91.7%`).

Board timestamps in this refresh span `2026-08-20T20:59:07Z` to `2026-08-21T11:27:51Z` (board metadata).

| Group | Seeds done | Notes |
| --- | ---: | --- |
| `none` | `3` / `3` | complete |
| `add11_del01` | `3` / `3` | complete |
| `add15_del05` | `2` / `3` | seeds `100` and `102` done; seed `101` stuck `running` (stale folder skipped by driver) |
| `add20_del09` | `3` / `3` | complete |

To finish the last cell, delete the stale folder and re-run the driver:

```text
experiments/output/train_mnist/runs/exp006_neuron_resize_actions/add15_del05/.../seed_101
```

## Why this experiment

Experiments 001–005 kept AddNeurons / DelNeurons off while layer resize was unstable. After the layer-resize fix, `RunningConfig` enables all six neuron-resize flags by default. Exp 001–005 still force them off locally so old grids stay comparable.

This short probe asks whether those six flags deserve to stay on for new runs. We test them as three paired ratio groups against a no-resize control, instead of enabling all six at once.

## Measurements and charts

Charts below use the `11` completed boards only.

Generate / refresh:

```text
python documentation/website/scripts/generate_experiment_006_charts.py
```

### Final accuracy by group

Mean final validation (completed seeds):

| Group | Seeds | Mean val (%) |
| --- | ---: | ---: |
| `none` | `3` | `84.52` |
| `add11_del01` | `3` | `82.64` |
| `add15_del05` | `2` | `84.76` |
| `add20_del09` | `3` | `84.60` |

![Final accuracy by neuron-resize group](/assets/experiments/006-final-accuracy-by-group.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy by neuron-resize group. Gray markers are individual completed seeds.

On full three-seed groups, `add20_del09` is essentially tied with `none` (`84.60` vs `84.52`). The mild pair is worse. The medium pair looks slightly high on two seeds, but seed `101` is still missing.

### Parameter growth by group

All completed runs start at `420` params. Mean finals: `none` `2080`, `add11_del01` `2407`, `add15_del05` `2776` (two seeds), `add20_del09` `2088`.

![Parameter growth by neuron-resize group](/assets/experiments/006-param-growth-by-group.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by group. Gray markers are individual final counts.

Growth comes from residual and sequential layer adds. It is not evidence that width-resize actions ran.

### Chosen simulation actions

Across `56` simulation calls on completed runs, the winner was always a layer-structure action:

| Chosen short label | Count |
| --- | ---: |
| `Add Res Conv Layer Action` | `40` |
| `Add Seq Linear Layer Action` | `9` |
| `Add Seq Conv Layer Action` | `4` |
| `Add Seq Dropout Layer Action` | `2` |
| `Add Res Linear Layer Action` | `1` |
| `Add Neurons` / `Delete Neurons` | `0` |

![Chosen simulation actions by neuron-resize group](/assets/experiments/006-simulation-chosen-actions-by-group.png)

> [!CAPTION] Figure 3. Count of winning simulation actions by group. Residual convolution wins most calls.

![Executed action mix by neuron-resize group](/assets/experiments/006-action-composition-by-group.png)

> [!CAPTION] Figure 4. Mean executed live actions by short label and group after training.

### Were AddNeurons / DelNeurons even possible?

Yes. Candidate lists in `simulation_gen_*.json` show neuron-resize actions in the root pool whenever the matching flags were on. They were almost never scored, and never chosen.

| Group | Sims | Sims with neuron in pool | Sims with neuron scored | Neuron entries in pool | Scored | Unscored (`visits=0`) | Chosen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | `15` | `0` | `0` | `0` | `0` | `0` | `0` |
| `add11_del01` | `15` | `15` | `2` | `15` Add / `0` Del | `2` | `13` | `0` |
| `add15_del05` | `10` | `10` | `0` | `11` Add / `11` Del | `0` | `22` | `0` |
| `add20_del09` | `16` | `16` | `0` | `19` Add / `16` Del | `0` | `35` | `0` |

![Neuron-resize presence vs scoring in simulation](/assets/experiments/006-neuron-candidate-scoring.png)

> [!CAPTION] Figure 5. For each group: how many simulations listed a neuron-resize candidate, and how many actually scored one.

Reading notes:

1. In enabled groups, neuron actions were in the candidate pool in `100%` of recorded simulations (`15/15`, `10/10`, `16/16`).
2. `DelNeurons` with ratio `0.1` never appears in the pool (`add11_del01` del count `0`). That matches `MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 5`: on width `16`, `int(16*0.1)=1` is illegal.
3. Most neuron candidates stay at `visits=0` and `score=null`. `sequential_halving_beam` scores root arms in registry order. Neuron actions are registered after residual/seq/dropout actions, so the `120 s` root budget often ends before they are scored.
4. Only `2` scored neuron candidates appear in the whole completed grid, both `AddNeurons@1.1` on `add11_del01` seed `100`.

### Grades and how close neuron actions were

Composite simulation scores come from `SimulationScore` (`val_acc` term weight `1.0`, param-count weight `0.1`). Higher is better.

The only scored near-misses:

| Seed | Gen | Candidate | Score | Rank among scored | Winner | Winner score | Gap to best |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `100` | `3` | `AddNeurons@1.1` | `0.7454` | `7` / `15` | `Add Res Conv Layer Action` | `0.7617` | `0.0164` |
| `100` | `0` | `AddNeurons@1.1` | `0.2363` | `14` / `14` | `Add Res Conv Layer Action` | `0.3199` | `0.0836` |

So when neuron-resize was actually graded, the closest miss was about `0.016` composite points behind a residual-conv winner. It was not near first place (`rank 7`). The other scored case was last among scored arms.

For `add15_del05` and `add20_del09`, every neuron candidate in the pool was unscored (`visits=0`). Those groups never got a fair graded comparison inside Sequential Halving.

### Training curves by group

![Training accuracy curves by neuron-resize group](/assets/experiments/006-training-curves.png)

> [!CAPTION] Figure 6. Training accuracy over epochs for every completed seed, colored by group.

Curves overlap by group. Seed variance is larger than most group gaps.

## Grouped final results

Completed cells only.

| Group | Seeds | Mean train (%) | Mean val (%) | Mean final params | Neuron-resize actions used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | `3` | `82.41` | `84.52` | `2080` | `0` |
| `add11_del01` | `3` | `83.80` | `82.64` | `2407` | `0` |
| `add15_del05` | `2` | `82.16` | `84.76` | `2776` | `0` |
| `add20_del09` | `3` | `82.78` | `84.60` | `2088` | `0` |

Per-seed validation for completed cells:

| Group | seed `100` val (%) | seed `101` val (%) | seed `102` val (%) |
| --- | ---: | ---: | ---: |
| `none` | `83.38` | `88.97` | `81.22` |
| `add11_del01` | `74.00` | `89.88` | `84.04` |
| `add15_del05` | `82.33` |  | `87.19` |
| `add20_del09` | `78.45` | `92.22` | `83.14` |

Matched two-seed check for `add15_del05` (seeds `100` and `102` only): mean val `84.76` versus `none` on the same seeds `82.30`. That looks helpful, but it is only two seeds and the missing seed prevents a full matched comparison.

## Training-history analysis

Completed runs usually execute five architecture actions over eight generations. Most actions are residual convolution inserts. Training accuracy often jumps after the first residual add, then climbs more slowly.

The mild pair has one weak seed (`add11_del01` seed `100`, val `74.00%`). The aggressive pair has one very strong seed (`add20_del09` seed `101`, val `92.22%`). Those seed swings are larger than the mean gaps between groups.

Because no neuron-resize action was executed, group differences are not caused by width change. They come from ordinary seed noise in layer-add search, plus a larger unused candidate list that often never gets scored.

## Limitations and seed effects

- One cell is still incomplete: stale `add15_del05` seed `101`.
- Short run (`64` epochs) can understate late width changes if those actions ever get selected.
- Three seeds are enough for a first ranking, not for a hard reject of a close second place.
- MNIST `big` may need few growth steps, so residual adds dominate the legal move list.
- Neuron-resize candidates are registered late in `generate_all_actions`. Under a finite Sequential Halving budget they are often left at `visits=0`, so “enabled” does not mean “graded.”
- `DEL_NEURONS_01` is effectively dead on this starter width because of the minimum matrix size rule.

## Conclusions

Based on the completed cells and simulation candidate files in this refresh:

1. Neuron-resize actions were legal in the simulation pool for every enabled-group call (`41/41` sims with flags on), but they were chosen `0` times.
2. They were almost never graded: only `2` scored AddNeurons events in `56` simulations. The closest scored miss was `0.016` behind a residual-conv winner (`rank 7/15`).
3. On full three-seed groups, `none` and `add20_del09` are effectively tied on mean validation (`84.52%` vs `84.60%`). The mild pair is worse (`82.64%`).
4. Enabling the flags on this package mostly adds unscored candidates and does not change the executed action family. Prefer keeping neuron-resize off for Exp 001–005-style runs until a probe both scores and selects them.

## Next experiments

1. Delete the stale `add15_del05` seed `101` folder and finish that one cell, then refresh this page to `12` / `12`.
2. Re-run with neuron-resize registered earlier, or with a longer root Sequential Halving budget, so AddNeurons / DelNeurons get `visits>0` before elimination.
3. If graded neuron actions still lose by a clear score gap, turn the six `ACTIONS_ENABLE_*_NEURONS_*` defaults back off for normal training.
4. Re-test only one candidate pair on a harder starter or dataset where width change is more likely to be useful.
