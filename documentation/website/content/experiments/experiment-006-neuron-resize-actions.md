# Experiment 006: Neuron-resize action ratio pairs

We keep the Experiment 005 / train-ci package (`sequential_halving_beam` + `composed_exponential` + `big` starter). We change only which AddNeurons / DelNeurons ratio pair is enabled.

The goal is to learn whether the six neuron-resize flags in `RunningConfig` are useful on a short MNIST probe, or whether search ignores them and they should stay off.

This refresh is the rerun after the keep-set fix in `sequential_halving_beam_alg.get_action` (`SIMULATION_MIN_ALGORITHM_ITERATION_RUNS = 3` and sort-by-mean before `BEAM_WIDTH`).

Script: `experiments/train_mnist_exp006_neuron_resize_actions.py`

Charts: `documentation/website/scripts/generate_experiment_006_charts.py`

Folder: `experiments/output/train_mnist/runs/exp006_neuron_resize_actions`

Snapshot: `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`

Simulation-candidate analysis: `documentation/website/data/experiments/experiment-006-simulation-action-analysis.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`12` / `12` cells, `100.0%`). Simulation tables use `board/simulations/simulation_gen_*.json` from those runs (`60` simulation calls).

The 0% progress line is not this grid. Each cell writes `board/main.json` under `exp006_neuron_resize_actions/<group>/<hp_folder>/seed_<seed>/`. All twelve of those files have `status=completed`. Charts are built from those boards, not from an empty folder.

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
4. After the keep-set fix, is the chosen arm the highest scored root, not registry order?
5. Should default config keep all three pairs, keep one pair, or keep none?

## Result timeline

Progress in this refresh: `12` / `12` completed (`100.0%`).

Board timestamps in this refresh span `2026-08-23T17:43:42Z` to `2026-08-24T01:27:48Z` (board metadata: `experimentStartedOn` and `lastUpdate`). Mean wall time per cell is about `39` minutes.

| Group | Seeds done | Notes |
| --- | ---: | --- |
| `none` | `3` / `3` | complete |
| `add11_del01` | `3` / `3` | complete |
| `add15_del05` | `3` / `3` | complete |
| `add20_del09` | `3` / `3` | complete |

The earlier `2026-08-21` grid used a broken keep-set (registry order after a first-pass overrun). That grid is not mixed into these tables.

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
| `none` | `3` | `86.12` |
| `add11_del01` | `3` | `82.92` |
| `add15_del05` | `3` | `84.01` |
| `add20_del09` | `3` | `86.26` |

![Final accuracy by neuron-resize group](/assets/experiments/006-final-accuracy-by-group.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy by neuron-resize group. Gray markers are individual seeds.

`add20_del09` has the highest mean validation (`86.26%`), then `none` (`86.12%`). The gap is `0.14` percentage points. `add15_del05` (`84.01%`) and `add11_del01` (`82.92%`) sit below the control. Seed spread inside `none` is large (`75.82%` to `94.19%`). `add20_del09` executed zero neuron-resize actions, so its small lead over `none` is not a width-resize result.

### Parameter growth by group

All runs start at `420` params. Mean finals: `none` `1631`, `add11_del01` `1444`, `add15_del05` `1839`, `add20_del09` `2252`.

![Parameter growth by neuron-resize group](/assets/experiments/006-param-growth-by-group.png)

> [!CAPTION] Figure 2. Mean start and final parameter counts by group. Gray markers are individual final counts.

Most growth is from layer inserts. The three AddNeurons executions change linear width, so they also move final size. `add20_del09` ends largest without using a neuron-resize action.

### Did the keep-set fix change who wins?

Yes. In all `60` simulations, the chosen arm is the highest scored root candidate (`60` / `60` argmax). Candidate `visits` are `2` to `4` in every call (first pass plus Sequential Halving). `maxDepth` stays `1` in every call: the required halving rounds overrun the `120 s` budget, so beam deepen never starts. Mean simulation duration is `271 s` (range `121 s` to `634 s`).

Chosen winners across `60` simulations:

| Chosen short label | Count |
| --- | ---: |
| `Add Res Conv Layer Action` | `29` |
| `Add Seq Conv Layer Action` | `12` |
| `Add Seq Linear Layer Action` | `8` |
| `Add Seq Dropout Layer Action` | `5` |
| `Add Neurons Action` | `3` |
| `Add Res Linear Layer Action` | `2` |
| `Delete Layer Action` | `1` |
| `Delete Neurons Action` | `0` |

![Chosen simulation actions by neuron-resize group](/assets/experiments/006-simulation-chosen-actions-by-group.png)

> [!CAPTION] Figure 3. Count of winning simulation actions by group. Residual convolution is still the mode, but it is no longer the only winner.

On the `2026-08-21` grid every winner was residual convolution (`60` / `60`), and `38` / `60` chosen arms were not the global argmax. That lock is gone.

![Executed action mix by neuron-resize group](/assets/experiments/006-action-composition-by-group.png)

> [!CAPTION] Figure 4. Mean executed live actions by short label and group after training.

Every completed run executed five live actions (generations `0`, `3`, `4`, `5`, `6`). That is `60` live actions in total (`12` runs × `5`).

### How many times each live action ran, and the accuracy change

This section counts executed training actions, not simulation candidates. `add11` and `del01` are counted as their own rows, not lumped into “Add Neurons” / “Delete Neurons”.

How the accuracy change is measured, in percentage points:

1. Immediate: accuracy at the first epoch after the action, minus accuracy at the last epoch before it.
2. Recovered: accuracy at the end of the next generation (`8` epochs later), minus the same pre-action point.

Negative values mean accuracy fell. These are observational changes around that mutation. They mix the action with ordinary training and LR recovery.

Exact live counts (`60` actions):

| Live action | Times run |
| --- | ---: |
| `Add Res Conv Layer Action` | `29` |
| `Add Seq Conv Layer Action` | `12` |
| `Add Seq Linear Layer Action` | `8` |
| `Add Seq Dropout Layer Action` | `5` |
| `Add Res Linear Layer Action` | `2` |
| `Delete Layer Action` | `1` |
| `add11` (`AddNeurons@1.1`) | `1` |
| `add15` (`AddNeurons@1.5`) | `2` |
| `add20` (`AddNeurons@2.0`) | `0` |
| `del01` (`DelNeurons@0.1`) | `0` |
| `del05` (`DelNeurons@0.5`) | `0` |
| `del09` (`DelNeurons@0.9`) | `0` |

Mean accuracy change by live action. Residual conv is split because the generation-`0` insert (every run) is a jump from chance-level accuracy.

| Live action | n | Mean train immediate | Mean train recovered | Mean val immediate | Mean val recovered |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Add Res Conv Layer Action` (all) | `29` | `+1.35` | `+13.40` | `+2.68` | `+15.33` |
| `Add Res Conv Layer Action` (gen `0` only) | `12` | `+1.63` | `+28.57` | `+2.89` | `+32.54` |
| `Add Res Conv Layer Action` (later only) | `17` | `+1.15` | `+2.70` | `+2.52` | `+3.19` |
| `Add Seq Conv Layer Action` | `12` | `+1.16` | `+2.15` | `+1.86` | `+1.94` |
| `Add Seq Linear Layer Action` | `8` | `+0.78` | `+1.03` | `+1.23` | `+0.93` |
| `Add Seq Dropout Layer Action` | `5` | `-1.22` | `+0.69` | `+1.86` | `+1.80` |
| `Add Res Linear Layer Action` | `2` | `+0.46` | `+0.48` | `+0.94` | `+1.23` |
| `Delete Layer Action` | `1` | `-0.56` | `+2.15` | `+0.26` | `+2.65` |
| `add11` | `1` | `-1.84` | `+2.00` | `+1.14` | `+2.14` |
| `add15` | `2` | `+0.24` | `+1.07` | `+1.00` | `+1.50` |
| `add20` | `0` |  |  |  |  |
| `del01` | `0` |  |  |  |  |
| `del05` | `0` |  |  |  |  |
| `del09` | `0` |  |  |  |  |

`add11` dropped train immediately (`-1.84`) then recovered (`+2.00`). After the first residual-conv insert, later layer actions move training accuracy by a few percentage points.

### All six neuron-resize flags, counted separately

Each flag is counted on its own. Simulation counts are root candidates in `simulation_gen_*.json`. Live counts are executed training actions. Accuracy change uses the same immediate / recovered definition as above.

| Flag | Group where it can appear | In pool | Scored | Chosen / live | Mean sim score | Closest gap to best | Train imm. | Train rec. | Val imm. | Val rec. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `add11` | `add11_del01` | `13` | `13` | `1` | `0.6483` | `0.0000` (won) | `-1.84` | `+2.00` | `+1.14` | `+2.14` |
| `add15` | `add15_del05` | `16` | `16` | `2` | `0.6885` | `0.0000` (won twice) | `+0.24` | `+1.07` | `+1.00` | `+1.50` |
| `add20` | `add20_del09` | `17` | `17` | `0` | `0.6921` | `0.0028` |  |  |  |  |
| `del01` | `add11_del01` | `0` | `0` | `0` |  |  |  |  |  |  |
| `del05` | `add15_del05` | `15` | `15` | `0` | `0.5639` | `0.0718` |  |  |  |  |
| `del09` | `add20_del09` | `15` | `15` | `0` | `0.6590` | `0.0222` |  |  |  |  |

`add15` has `16` scored entries in `15` simulations because one call also listed `seq_linear_0` after a sequential linear insert. `add20` has `17` entries in `15` simulations for the same reason. `add11` never listed a second module.

Every listed entry was scored. `del01` never entered the pool: `MINIMUM_MATRIX_SIZE_FOR_NEURONS_REMOVAL = 5` rejects `int(16*0.1)=1`. `del05` and `del09` were legal and scored in every simulation of their groups. They never had the top root score. Closest `del05` gap is `0.0718`. Closest `del09` gap is `0.0222`. Closest `add20` gap is `0.0028` (`AddNeurons@2.0` `0.8629` vs residual conv `0.8657`, seed `101`, generation `4`).

Live events with accuracy (empty flags above never ran):

| Flag | Times run | Where | Train immediate | Train recovered | Val immediate | Val recovered |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `add11` | `1` | `add11_del01` seed `100`, gen `3` (epoch `32`), target `linear` | `-1.84` (`69.33` → `67.50`) | `+2.00` (to `71.34`) | `+1.14` (`69.98` → `71.12`) | `+2.14` (to `72.12`) |
| `add15` | `1` of `2` | `add15_del05` seed `101`, gen `5` (epoch `48`), target `linear` | `+0.02` (`81.21` → `81.23`) | `+0.68` (to `81.88`) | `+2.71` (`80.77` → `83.48`) | `+2.80` (to `83.57`) |
| `add15` | `1` of `2` | `add15_del05` seed `102`, gen `6` (epoch `56`), target `linear` | `+0.45` (`79.48` → `79.93`) | `+1.46` (to `80.94`) | `-0.71` (`83.24` → `82.53`) | `+0.20` (to `83.44`) |
| `add20` | `0` | never executed |  |  |  |  |
| `del01` | `0` | never in the pool |  |  |  |  |
| `del05` | `0` | scored `15` times, never chosen |  |  |  |  |
| `del09` | `0` | scored `15` times, never chosen |  |  |  |  |

### Were AddNeurons / DelNeurons possible, scored, and chosen?

Yes, when the matching flags are on. The six flags are not equal.

| Flag | Sims in that group | Sims with this flag in pool | Scored entries | Chosen |
| --- | ---: | ---: | ---: | ---: |
| `add11` | `15` | `13` | `13` | `1` |
| `add15` | `15` | `15` | `16` | `2` |
| `add20` | `15` | `15` | `17` | `0` |
| `del01` | `15` | `0` | `0` | `0` |
| `del05` | `15` | `15` | `15` | `0` |
| `del09` | `15` | `15` | `15` | `0` |

Two `add11_del01` seed `101` calls (generations `5` and `6`) have no `add11` candidate after earlier graph edits. `AddNeurons.generate_all_actions` targets hidden `nn.Linear` modules, so extra `seq_linear_*` rows appear only after a sequential linear insert.

![Neuron-resize presence vs scoring in simulation](/assets/experiments/006-neuron-candidate-scoring.png)

> [!CAPTION] Figure 5. For each group: how many simulations listed a neuron-resize candidate, and how many actually scored one.

Every listed neuron-resize entry was scored. The lumped Add/Del family means below still hide the ratio split: `add20` has the highest AddNeurons mean (`0.6921`) but never won; `del05` is the weakest scored family (`0.5639`).

### Mean score of each scored action

The value is the composite `SimulationScore` (`val_acc` weight `1.0`, param-count weight `0.1`). Higher is better. Each mean is over every scored root candidate of that family, not only winners.

Overall ranking across all `60` simulations:

| Action | Scored n | Mean score |
| --- | ---: | ---: |
| `Add Res Linear Layer Action` | `30` | `0.8083` |
| `Add Res Conv Layer Action` | `266` | `0.7324` |
| `Add Seq Linear Layer Action` | `202` | `0.7206` |
| `Add Seq Conv Layer Action` | `138` | `0.6955` |
| `Add Seq Dropout Layer Action` | `438` | `0.6810` |
| `Add Neurons Action` | `46` | `0.6785` |
| `Delete Layer Action` | `145` | `0.6634` |
| `Delete Neurons Action` | `30` | `0.6115` |

Mean score by group. Empty cells mean that family was not in the pool.

| Action | `none` | `add11_del01` | `add15_del05` | `add20_del09` |
| --- | ---: | ---: | ---: | ---: |
| `Add Neurons Action` |  | `0.6483` (`n=13`) | `0.6885` (`n=16`) | `0.6921` (`n=17`) |
| `Delete Neurons Action` |  |  | `0.5639` (`n=15`) | `0.6590` (`n=15`) |
| `Add Res Conv Layer Action` | `0.7367` (`n=69`) | `0.7202` (`n=59`) | `0.7325` (`n=72`) | `0.7387` (`n=66`) |
| `Add Res Linear Layer Action` | `0.7887` (`n=6`) | `0.8157` (`n=18`) | `0.8372` (`n=2`) | `0.7898` (`n=4`) |
| `Add Seq Conv Layer Action` | `0.7011` (`n=39`) | `0.6855` (`n=33`) | `0.6711` (`n=27`) | `0.7153` (`n=39`) |
| `Add Seq Linear Layer Action` | `0.7266` (`n=55`) | `0.7141` (`n=50`) | `0.7050` (`n=43`) | `0.7330` (`n=54`) |
| `Add Seq Dropout Layer Action` | `0.6855` (`n=114`) | `0.6833` (`n=111`) | `0.6650` (`n=99`) | `0.6883` (`n=114`) |
| `Delete Layer Action` | `0.7071` (`n=41`) | `0.6287` (`n=33`) | `0.5680` (`n=29`) | `0.7138` (`n=42`) |

![Mean simulation score by action](/assets/experiments/006-mean-simulation-score-by-action.png)

> [!CAPTION] Figure 6. Mean composite SimulationScore of every scored root candidate, by action family and neuron-resize group. Missing bars are families that were not in that group’s pool.

AddNeurons sits below residual conv on the family mean in every enabled group. It can still win a call when it is the top score that generation. DeleteNeurons is weaker than AddNeurons (`0.6115` vs `0.6785`) and was never chosen. Residual linear has the highest mean (`0.8083`) but a small pool (`n=30`).

### Training curves by group

![Training accuracy curves by neuron-resize group](/assets/experiments/006-training-curves.png)

> [!CAPTION] Figure 7. Training accuracy over epochs for every seed, colored by group.

Every run inserts residual convolution in generation `0` (epochs `0`–`8`). Generations `1` and `2` have no architecture action. The next four actions sit at generations `3`–`6`. One seed per several groups climbs far above the others (`none` `101`, `add11_del01` `101`, `add20_del09` `101`). The three AddNeurons executions are late (generations `3`, `5`, and `6`), so they do not create the first jump.

## Grouped final results

Completed cells only.

| Group | Seeds | Mean train (%) | Mean val (%) | Mean final params | `add*` used | `del*` used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `none` | `3` | `83.81` | `86.12` | `1631` | `0` | `0` |
| `add11_del01` | `3` | `79.84` | `82.92` | `1444` | `add11` × `1` | `del01` × `0` |
| `add15_del05` | `3` | `82.07` | `84.01` | `1839` | `add15` × `2` | `del05` × `0` |
| `add20_del09` | `3` | `84.66` | `86.26` | `2252` | `add20` × `0` | `del09` × `0` |

Per-seed validation:

| Group | seed `100` val (%) | seed `101` val (%) | seed `102` val (%) |
| --- | ---: | ---: | ---: |
| `none` | `75.82` | `94.19` | `88.36` |
| `add11_del01` | `78.03` | `91.47` | `79.26` |
| `add15_del05` | `84.81` | `83.77` | `83.44` |
| `add20_del09` | `78.46` | `92.03` | `88.30` |

Matched three-seed check: no enabled pair beats `none` on every seed. `add11_del01` and `add15_del05` beat `none` only on seed `100`. `add20_del09` beats `none` on seed `100`, loses on seed `101`, and is almost tied on seed `102` (`88.30` vs `88.36`). The control’s weak seed is `none` seed `100` (`75.82%` val). The strongest cell is `none` seed `101` (`94.19%` val).

## Training-history analysis

Completed runs execute five architecture actions over eight generations. Action generations are `0`, `3`, `4`, `5`, `6` (epoch windows starting at `0`, `24`, `32`, `40`, `48`).

The first residual-conv insert is shared. Later actions now differ by group and seed: sequential conv, sequential linear, dropout, residual linear, one layer delete, and three AddNeurons. That mix did not exist in the `2026-08-21` grid, where every live action was residual conv.

`add20_del09` stays on residual and sequential layer adds. Search scored AddNeurons@`2.0` in all `15` of those simulations and still preferred another family each time.

Because `maxDepth` stayed `1`, search is ranking root arms only. Group differences can come from that root ranking, from seed variance, and from the few width changes, not from look-ahead deepen.

## Limitations and seed effects

- Three seeds still show large spread (`none` validation `75.82%` to `94.19%`).
- Short run (`64` epochs) can understate late width changes.
- Sequential Halving now runs (`visits` `2`–`4`), but beam deepen never starts (`maxDepth = 1`).
- `configure_deterministic_seeding()` runs once at driver start. Cells then run in one process, so simulation gradient-descent noise is not a fresh matched draw per group.
- `DEL_NEURONS_01` is still dead on this starter width because of the minimum matrix size rule.

## How the report is preserved

The raw `experiments/output/` folder is ignored by Git. The report keeps:

- this Markdown page
- generated PNG charts under `documentation/website/app/public/assets/experiments/`
- a normalized run snapshot at `documentation/website/data/experiments/experiment-006-neuron-resize-actions.json`
- a simulation-candidate snapshot at `documentation/website/data/experiments/experiment-006-simulation-action-analysis.json`

`generate_experiment_006_charts.py` updates both snapshots when raw output exists. If raw output is missing, it reads the run snapshot instead.

These documentation artifacts must be committed before the raw experiment folder is removed from this machine.

## Conclusions

Based on the `12` completed cells and `60` simulation files in this refresh:

1. The keep-set fix works. Every simulation chose the highest scored root arm (`60` / `60`). Sequential Halving ran (`visits` `2`–`4`). The old residual-conv registry lock is gone.
2. Live neuron-resize counts: `add11` × `1`, `add15` × `2`, `add20` × `0`, `del01` × `0`, `del05` × `0`, `del09` × `0`. `del01` never entered the pool. `del05` and `del09` were scored in all `15` simulations of their groups and never had the top score. `add20` was scored `17` times and never won; closest gap `0.0028`.
3. Residual convolution is still the most common winner (`29` / `60`), but sequential conv/linear, dropout, residual linear, and delete-layer also win when they have the top score.
4. Family-mean AddNeurons (`0.6785`) remains below residual conv (`0.7324`). Winning is about being best in that call, not about having the highest family average.
5. Mean validation is `add20_del09` `86.26%`, `none` `86.12%`, `add15_del05` `84.01%`, `add11_del01` `82.92%`. The `add20_del09` lead is not from width change (`0` neuron-resize live actions). No pair beats `none` on every matched seed. That is not a reason to turn the six flags on by default for train-ci.

## Next experiments

1. Keep the six `ACTIONS_ENABLE_*_NEURONS_*` flags off in Exp 001–005-style and train-ci packages, unless a harder task shows a clear matched-seed gain.
2. If beam deepen should run, raise `simulation_time` or cut the first-pass cost. Required Sequential Halving already overruns `120 s`.
3. Drop or replace `DEL_NEURONS_01` on this starter. The minimum matrix size rule makes ratio `0.1` illegal on width `16`.
4. Re-test one grow ratio on a harder starter or dataset if width change is the research target. On this MNIST `big` probe, layer inserts still dominate even when ranking is fair.
