# Experiment 006: Neuron-resize action ratio pairs

We keep the Experiment 005 / train-ci package (`sequential_halving_beam` + `composed_exponential` + `big` starter). We change only which AddNeurons / DelNeurons ratio pair is enabled.

The goal is to learn whether the six neuron-resize flags in `RunningConfig` are useful on a short MNIST probe, or whether search ignores them and they should stay off.

Script: `experiments/train_mnist_exp006_neuron_resize_actions.py`

Charts: `documentation/website/scripts/generate_experiment_006_charts.py`

Folder: `experiments/output/train_mnist/runs/exp006_neuron_resize_actions`

Snapshot: `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`

Simulation-candidate analysis: `documentation/website/data/experiments/experiment-006-simulation-action-analysis.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`12` / `12` cells, `100.0%`). Simulation tables use `board/simulations/simulation_gen_*.json` from those runs (`60` simulation calls).

The main simulation summary is the mean composite `SimulationScore` of each scored action family. Pool presence is not enough. In this grid every enabled neuron-resize candidate was listed and scored.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Neuron-resize group | `none`, `add11_del01`, `add15_del05`, `add20_del09` | Compare no width change vs mild / medium / aggressive ratio pairs |
| Seed | `100`, `101`, `102` | Three matched seeds per group |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `12` | `4` groups × `3` seeds |
| Completed cells in this refresh | `12` | Full grid |
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
3. When they are scored, how high is their mean `SimulationScore` next to other action families?
4. Does enabling a pair grow or shrink parameter count in a useful way?
5. Should default config keep all three pairs, keep one pair, or keep none?

## Result timeline

Progress in this refresh: `12` / `12` completed (`100.0%`).

Board timestamps in this refresh span `2026-08-21T15:12:08Z` to `2026-08-21T21:56:12Z` (board metadata: `experimentStartedOn` and `lastUpdate`). Mean wall time per cell is about `33` minutes.

| Group | Seeds done | Notes |
| --- | ---: | --- |
| `none` | `3` / `3` | complete |
| `add11_del01` | `3` / `3` | complete |
| `add15_del05` | `3` / `3` | complete |
| `add20_del09` | `3` / `3` | complete |

## Why this experiment

Experiments 001–005 kept AddNeurons / DelNeurons off while layer resize was unstable. After the layer-resize fix, `RunningConfig` enables all six neuron-resize flags by default. Exp 001–005 still force them off locally so old grids stay comparable.

This short probe asks whether those six flags deserve to stay on for new runs. We test them as three paired ratio groups against a no-resize control, instead of enabling all six at once.

## Measurements and charts

Charts below use the `12` completed boards.

Generate / refresh:

```text
python documentation/website/scripts/generate_experiment_006_charts.py
```

### Final accuracy by group

Mean final validation (all seeds):

| Group | Seeds | Mean val (%) |
| --- | ---: | ---: |
| `none` | `3` | `82.85` |
| `add11_del01` | `3` | `88.72` |
| `add15_del05` | `3` | `85.13` |
| `add20_del09` | `3` | `89.61` |

![Final accuracy by neuron-resize group](/assets/experiments/006-final-accuracy-by-group.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy by neuron-resize group. Gray markers are individual seeds.

`add20_del09` has the highest mean validation (`89.61%`). `add11_del01` is next (`88.72%`). Both beat `none` on every matched seed. `add15_del05` is in between and overlaps `none`. These gaps are not from executed width change. See the action and simulation sections.

### Parameter growth by group

All runs start at `420` params. Mean finals: `none` `2048`, `add11_del01` `3084`, `add15_del05` `2788`, `add20_del09` `2788`.

![Parameter growth by neuron-resize group](/assets/experiments/006-param-growth-by-group.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by group. Gray markers are individual final counts.

Growth comes from residual convolution inserts at different graph sites. It is not evidence that width-resize actions ran.

### Chosen simulation actions

Across `60` simulation calls, the winner was always residual convolution:

| Chosen short label | Count |
| --- | ---: |
| `Add Res Conv Layer Action` | `60` |
| `Add Neurons` / `Delete Neurons` | `0` |
| any other family | `0` |

![Chosen simulation actions by neuron-resize group](/assets/experiments/006-simulation-chosen-actions-by-group.png)

> [!CAPTION] Figure 3. Count of winning simulation actions by group. Every winner is residual convolution.

![Executed action mix by neuron-resize group](/assets/experiments/006-action-composition-by-group.png)

> [!CAPTION] Figure 4. Mean executed live actions by short label and group after training.

Every completed run executed five live actions. Every live action was `Add Res Conv Layer Action`.

### Were AddNeurons / DelNeurons even possible?

Yes. Candidate lists in `simulation_gen_*.json` show neuron-resize actions in the root pool whenever the matching flags were on. In this refresh they were always scored (`visits=1`). They were never chosen.

| Group | Sims | Sims with neuron in pool | Sims with neuron scored | Neuron entries in pool | Scored | Unscored (`visits=0`) | Chosen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | `15` | `0` | `0` | `0` | `0` | `0` | `0` |
| `add11_del01` | `15` | `15` | `15` | `15` Add / `0` Del | `15` | `0` | `0` |
| `add15_del05` | `15` | `15` | `15` | `15` Add / `15` Del | `30` | `0` | `0` |
| `add20_del09` | `15` | `15` | `15` | `15` Add / `15` Del | `30` | `0` | `0` |

![Neuron-resize presence vs scoring in simulation](/assets/experiments/006-neuron-candidate-scoring.png)

> [!CAPTION] Figure 5. For each group: how many simulations listed a neuron-resize candidate, and how many actually scored one.

Enabled groups listed a neuron-resize candidate in `15/15` simulations each, and scored every one of those entries (`75` scored events, `0` unscored). `DelNeurons` with ratio `0.1` never appears (`add11_del01` del count `0`). That matches `MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 5`: on linear width `16`, `int(16*0.1)=1` is illegal. `AddNeurons.generate_all_actions` only targets `nn.Linear` hidden modules, so the scored grow action is `['linear', ratio]`.

### Mean score of each scored action

Figure 5 only shows that the families were listed and graded. The main summary is how high those grades were.

The value is the composite `SimulationScore` (`val_acc` weight `1.0`, param-count weight `0.1`). Higher is better. Each mean is over every scored root candidate of that family, not only winners.

Overall ranking across all `60` simulations:

| Action | Scored n | Mean score |
| --- | ---: | ---: |
| `Delete Layer Action` | `185` | `0.7228` |
| `Add Seq Linear Layer Action` | `221` | `0.7160` |
| `Add Seq Conv Layer Action` | `176` | `0.6981` |
| `Add Res Conv Layer Action` | `236` | `0.6971` |
| `Add Neurons Action` | `45` | `0.6725` |
| `Add Seq Dropout Layer Action` | `444` | `0.6614` |
| `Delete Neurons Action` | `30` | `0.5932` |

Mean score by group. Empty cells mean that family was not in the pool.

| Action | `none` | `add11_del01` | `add15_del05` | `add20_del09` |
| --- | ---: | ---: | ---: | ---: |
| `Add Neurons Action` |  | `0.6828` (`n=15`) | `0.6307` (`n=15`) | `0.7041` (`n=15`) |
| `Delete Neurons Action` |  |  | `0.4980` (`n=15`) | `0.6885` (`n=15`) |
| `Add Res Conv Layer Action` | `0.6733` (`n=73`) | `0.7187` (`n=47`) | `0.6689` (`n=61`) | `0.7414` (`n=55`) |
| `Add Seq Conv Layer Action` | `0.6759` (`n=58`) | `0.7171` (`n=32`) | `0.6694` (`n=46`) | `0.7483` (`n=40`) |
| `Add Seq Linear Layer Action` | `0.6849` (`n=58`) | `0.7477` (`n=52`) | `0.6747` (`n=55`) | `0.7591` (`n=56`) |
| `Add Seq Dropout Layer Action` | `0.6435` (`n=132`) | `0.6780` (`n=93`) | `0.6350` (`n=114`) | `0.6979` (`n=105`) |
| `Delete Layer Action` | `0.6817` (`n=49`) | `0.7600` (`n=43`) | `0.7041` (`n=47`) | `0.7511` (`n=46`) |

![Mean simulation score by action](/assets/experiments/006-mean-simulation-score-by-action.png)

> [!CAPTION] Figure 6. Mean composite SimulationScore of every scored root candidate, by action family and neuron-resize group. Missing bars are families that were not in that group’s pool.

AddNeurons sits below residual and sequential layer-add means in the same group:

| Group | AddNeurons mean | Residual conv mean |
| --- | ---: | ---: |
| `add11_del01` (`×1.1`) | `0.6828` | `0.7187` |
| `add15_del05` (`×1.5`) | `0.6307` | `0.6689` |
| `add20_del09` (`×2.0`) | `0.7041` | `0.7414` |

DeleteNeurons is weaker. Ratio `0.5` has the lowest family mean in the grid (`0.4980`). Ratio `0.9` is closer (`0.6885`) but still below residual conv in that group (`0.7414`). These are observational score averages. They do not say what would have happened if Sequential Halving had kept the top-scoring arms.

### Why high-scoring neuron actions still lost

Scoring is not the same as being eligible to win. `sequential_halving_beam_alg.get_action` first grades every root arm once. In all `60` calls that first pass ran past the `120 s` budget (recorded duration `120.3 s` to `407.7 s`). The Sequential Halving loop then never sorted or halved the living set.

The beam is `survivors[:BEAM_WIDTH]` with `BEAM_WIDTH = 3`. After a first pass that used up the root budget, that slice is registry order from `generate_all_actions`. Residual convolution is listed first. Neuron-resize flags are registered after residual, sequential, and dropout actions, so they never enter the keep set.

`14` / `60` calls reached `maxDepth = 2` (a few deepen rollouts). Those still only deepen the first three residual-conv roots. All `15` `add20_del09` calls stayed at `maxDepth = 1`.

The recorded candidate `score` is the root-arm mean after one visit. The chosen action is not the global argmax: `38` / `60` calls picked a residual-conv arm that was not the highest root score.

Closest scored neuron-resize misses:

| Group | Seed | Gen | Candidate | Score | Rank among scored | Winner score | Gap to best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `add20_del09` | `100` | `4` | `AddNeurons@2.0` | `0.8106` | `1` / `20` | `0.7944` | `0.0000` |
| `add20_del09` | `102` | `6` | `AddNeurons@2.0` | `0.8722` | `2` / `21` | `0.8597` | `0.0039` |
| `add15_del05` | `102` | `5` | `AddNeurons@1.5` | `0.8325` | `3` / `31` | `0.8143` | `0.0109` |

The strongest near-miss is not a near miss in the keep set. `AddNeurons@2.0` had the best root score in `add20_del09` seed `100` generation `4` and still lost, because the beam was the first three residual-conv arms. Mean neuron rank among scored arms is `12.5` (`add20_del09`), `12.5` (`add11_del01`), and `15.9` (`add15_del05`).

### Training curves by group

![Training accuracy curves by neuron-resize group](/assets/experiments/006-training-curves.png)

> [!CAPTION] Figure 7. Training accuracy over epochs for every seed, colored by group.

Most runs jump after the first residual add at global epoch `8`, then climb more slowly after later residual adds. Three runs stay near chance through generation `0` (`none` seeds `101` and `102`, `add15_del05` seed `101`) and jump after an extra action at epoch `16`. Seed variance is visible inside each color.

## Grouped final results

Completed cells only.

| Group | Seeds | Mean train (%) | Mean val (%) | Mean final params | Neuron-resize actions used |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` | `3` | `81.34` | `82.85` | `2048` | `0` |
| `add11_del01` | `3` | `86.37` | `88.72` | `3084` | `0` |
| `add15_del05` | `3` | `84.06` | `85.13` | `2788` | `0` |
| `add20_del09` | `3` | `87.37` | `89.61` | `2788` | `0` |

Per-seed validation:

| Group | seed `100` val (%) | seed `101` val (%) | seed `102` val (%) |
| --- | ---: | ---: | ---: |
| `none` | `85.28` | `84.88` | `78.40` |
| `add11_del01` | `86.04` | `91.49` | `88.63` |
| `add15_del05` | `86.86` | `84.40` | `84.13` |
| `add20_del09` | `88.36` | `93.53` | `86.93` |

Matched three-seed check: `add11_del01` and `add20_del09` beat `none` on every seed. `add15_del05` beats `none` on seeds `100` and `102`, and loses on seed `101` (`84.40` vs `84.88`). The control’s weak seed is `none` seed `102` (`78.40%` val, `1604` params). The strongest cell is `add20_del09` seed `101` (`93.53%` val).

## Training-history analysis

Completed runs execute five architecture actions over eight generations. The usual action epochs are `8`, `32`, `40`, `48`, `56`. Three runs insert at epoch `16` instead of `32`.

Training accuracy often jumps `13` to `30` percentage points in the generation after the first residual add that actually learns. Later residual adds add a few percentage points each. Curves then flatten toward the end of the `64` epochs.

Because no neuron-resize action was executed, group differences are not caused by width change. They come from different residual-conv insertion sites, different final sizes, and seed variance. Enabling extra root arms also lengthens the first scoring pass, which can cut beam deepen (`add20_del09` never reached `maxDepth = 2`). That can change which of the first three residual-conv arms is returned, without ever applying AddNeurons or DelNeurons.

## Limitations and seed effects

- Short run (`64` epochs) can understate late width changes if those actions ever get selected.
- Three seeds are enough for a first ranking, not for a hard default-on decision.
- MNIST `big` may need few growth steps, so residual adds dominate the legal move list.
- `sequential_halving_beam` did not run Sequential Halving on this budget. “Enabled and scored” does not mean “in the keep set.”
- `configure_deterministic_seeding()` runs once at driver start. Cells then run in one process, so simulation gradient-descent noise is not a fresh matched draw per group.
- `DEL_NEURONS_01` is effectively dead on this starter width because of the minimum matrix size rule.

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. The report keeps:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- a normalized run snapshot at `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`
- a simulation-candidate snapshot at `documentation/website/data/experiments/experiment-006-simulation-action-analysis.json`

`generate_experiment_006_charts.py` updates both snapshots when raw output exists. If raw output is missing, it reads the run snapshot instead.

These documentation artifacts must be committed before the raw experiment folder is removed from this machine.

## Conclusions

Based on the completed cells and simulation candidate files in this refresh:

1. Neuron-resize actions were legal and scored in every enabled-group call (`45/45` simulations with flags on, `75` scored neuron entries). They were chosen `0` times.
2. Mean AddNeurons score is `0.6725` (`n=45`). Mean DelNeurons score is `0.5932` (`n=30`). Mean residual-conv score is `0.6971` (`n=236`). In every enabled group, AddNeurons scored below residual conv. DelNeurons@`0.5` is the weakest family (`0.4980`).
3. Search never gave them a fair keep-set comparison. After a first pass that overran `120 s`, `sequential_halving_beam` kept the first three `generate_all_actions` arms. Those arms are residual convolution. One `AddNeurons@2.0` had the best root score (`0.8106` vs winner `0.7944`) and still lost.
4. On the three-seed means, `add20_del09` leads validation (`89.61%`), then `add11_del01` (`88.72%`), then `add15_del05` (`85.13%`), then `none` (`82.85%`). That ranking is not a width-resize result. Every live action was residual convolution.
5. Enabling the flags on this package adds scored candidates that cannot win under the current beam rule, and it can change residual-conv selection by using up simulation time. Prefer keeping neuron-resize off for Exp 001–005-style runs until Sequential Halving actually ranks them into the keep set.

## Next experiments

1. Change `sequential_halving_beam_alg.get_action` so `survivors[:BEAM_WIDTH]` is the top arms by mean even when the Sequential Halving loop never runs.
2. Re-run this `12`-cell grid after that change, so a scored AddNeurons / DelNeurons arm can win when it has the best root score.
3. If first-pass scoring still overruns `120 s`, raise the budget or score fewer root arms before halving, so Sequential Halving and beam deepen actually run.
4. If graded neuron actions then still lose by a clear score gap, turn the six `ACTIONS_ENABLE_*_NEURONS_*` defaults back off for normal training.
5. Re-test only one candidate pair on a harder starter or dataset where width change is more likely to be useful.
