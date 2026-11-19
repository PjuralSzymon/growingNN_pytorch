# Experiment 010: MCTS meta-parameter tuning

Experiment 010 studies simulation algorithms for GrowingNN in two stages. First, we compared two proposed variants of the existing Monte Carlo Tree Search (MCTS) algorithm with the Monte Carlo implementation already in use. Neither variant improved the results in the completed comparison, so we retained the current algorithm and focused this report on tuning its meta-parameters. This tuning study tests one parameter at a time, carries each selected value forward, and repeats the sequence in a second pass.

## Related experiment: MCTS variants

This comparison was run in a sibling repository on CIFAR-10. Its purpose was to decide whether a new simulation-algorithm variant justified further development, before investing effort in tuning the existing MCTS. It compared the current algorithm with stabilized and extended variants. Results were recorded for two seeds per method in each of two phases.

### Tested approaches

#### Existing Monte Carlo MCTS

The existing implementation was the baseline. It generates legal architecture mutations, expands a search tree, briefly trains candidate models, and uses UCB1 to choose which branch to explore. A rollout may apply randomly selected mutations before scoring the resulting model. The search returns a root mutation selected from the explored tree. The experiment treated this as the current approach to keep if neither alternative showed a clear improvement.

#### Stabilized MCTS

This variant aimed to make mutation choices more reliable by measuring the benefit of a change against ordinary training progress. It evaluates a sequence of network changes and compares the resulting model with an unchanged copy trained under the same conditions. This paired comparison helps separate the effect of the changes from variation caused by training. The most promising first changes are then checked again with new seeds before one is applied.

- Try sequences of network changes.
- Train each changed network and an unchanged copy with the same seed and for the same time.
- Score each sequence by how much its accuracy improves over the unchanged copy.
- Test the most promising first changes again with new seeds.
- Keep a change only if it still improves accuracy; otherwise, leave the network unchanged.

#### Extended MCTS

This variant aimed to look further ahead than a single mutation by testing short sequences of network changes. It begins with candidate changes, then gradually considers additional options while directing more of its search toward sequences that have scored well so far. Each sequence is compared with an unchanged model under matching training conditions. Before choosing which first change to apply, the strongest candidates are evaluated again with new seeds.

- Start with possible network changes and test short sequences of them.
- Add more options as the search continues, focusing on sequences that look promising.
- Compare each changed network with an unchanged copy, using the difference in accuracy as its score.
- Test the strongest first changes again with new seeds.
- Apply a change only if it still improves accuracy; otherwise, leave the network unchanged.

### Results and decision

Both phases started from a deliberately very small network so the simulation algorithms had room to add architecture components. This constrained starter and the short Phase 1 training setup help explain the low accuracies. Phase 2 used a longer schedule and stronger training recipe, with the same starter. Mean official test accuracies were:

| Phase | Existing Monte Carlo MCTS | Stabilized MCTS | Extended MCTS |
|---|---:|---:|---:|
| 1 | **58.80%** | 51.82% | 56.51% |
| 2 | **59.33%** | 55.94% | 55.53% |

Existing MCTS had the highest mean in both phases, so the study retained it and shifted effort to tuning its meta-parameters.

Results are preliminary: each mean uses two seeds, and the planned CIFAR-100 comparison was not available. The current tuning runs use a different starter and training setup, so they are not a direct benchmark against these results.

## Candidate values

Each candidate is evaluated on CIFAR-10 with seeds `1` and `2`: two runs per value.

| Order | Meta-parameter | Candidate values |
|---:|---|---|
| 1 | MCTS · UCB1 exploration constant (`MCTS_UCB1_C`) | `0.5`, `1`, `2` (default), `4`, `8` |
| 2 | MCTS · UCB1 formula (`MCTS_UCB1_USE_SQRT`) | `False` (default), `True` |
| 3 | MCTS · rollout value propagation (`MCTS_PROPAGATE_ROLLOUT_VALUE`) | `False` (default), `True` |
| 4 | MCTS · rollout depth (`MCTS_ROLLOUT_DEPTH`) | `0`, `1`, `2` (default), `3` |
| 5 | MCTS · rollout training epochs (`MCTS_ROLLOUT_EPOCHS`) | `1` (default), `2`, `4` |
| 6 | MCTS · rollout learning rate (`MCTS_ROLLOUT_LR`) | `0.0001` (configured default), `0.003`, `0.01` |
| 7 | Simulation · minimum iterations (`SIMULATION_MIN_ALGORITHM_ITERATION_RUNS`) | `1`, `3` (default), `5` |
| 8 | Simulation · time budget (`simulation_time`) | `8s` (default), `16s`, `32s` |
| 9 | Simulation · training-set percentage | Current 128-example request (`0.284%`), `5%`, `10%`, `30%`, `40%` |

The rollout learning-rate scheduler floors its applied rate at `0.001`, so the configured default `0.0001` already behaves like `0.001`; that duplicate candidate is omitted.

## Progress and runtime estimate

**Completed 19 November 2026 at 08:11 UTC (09:11 Warsaw):** all **120 of 120 runs** and all `24,000/24,000` epochs are complete. Total elapsed wall time was about **8 days 10 hours 12 minutes**.

The experiment used CIFAR-10 with the tiny starter; the earlier CIFAR-100 pilot was excluded from this run.

## Experiment setup

The initial CIFAR-10 graph follows the small MNIST starter pattern: one `3→8` channel convolution, ReLU, adaptive average pooling, and an `8→10` linear classifier. It has `296` parameters and no BatchNorm. The model, augmentation, and optimizer can then grow through the enabled architecture actions.

Training uses `10` generations × `20` epochs with a cosine learning rate from `0.1` to `0.001` over all `200` epochs. Seeds are `1` and `2`. Candidate selection uses mean final validation accuracy; official test results are reported for context, not used to select settings.

## Pass 1

### Meta-parameters

#### MCTS · UCB1 exploration constant (`MCTS_UCB1_C`)

The tested UCB1 constants rank by mean final validation accuracy as `1.0` (`56.28%`), `8.0` (`50.75%`), `2.0` (`49.85%`), `4.0` (`48.50%`), and `0.5` (`46.62%`). The runner therefore selected `1.0` over the default `2.0`. The result is provisional: there are only two seeds per value, and `C=1.0` has a `16.08`-point spread between its seeds.

![Training and validation histories by UCB1 C candidate](/assets/experiments/010-pass-1-ucb1-c-training-histories-grid.svg)

> [!CAPTION] Figure 1. Compact small multiples follow the training-history layout used in Experiment 002: one panel per tested C value, with shared epoch and accuracy scales. Solid lines show training accuracy, dashed lines show validation accuracy, and color identifies seed. Vertical dotted lines mark generation boundaries.

| UCB1 C | Mean final validation | Seed range | Mean final test | Mean run time |
|---:|---:|---:|---:|---:|
| `0.5` | `46.62%` | `41.36–51.88%` | `46.87%` | `21.12 min` |
| `1.0` **selected** | **`56.28%`** | `48.24–64.32%` | `56.26%` | `33.31 min` |
| `2.0` (default) | `49.85%` | `49.62–50.08%` | `49.60%` | `32.17 min` |
| `4.0` | `48.50%` | `40.44–56.56%` | `48.30%` | `31.00 min` |
| `8.0` | `50.75%` | `46.48–55.02%` | `50.80%` | `30.89 min` |

Per-seed results:

| UCB1 C | Seed | Final validation | Best validation | Final test | Runtime |
|---:|---:|---:|---:|---:|---:|
| `0.5` | `1` | `51.88%` | `52.64%` | `51.14%` | `20.35 min` |
| `0.5` | `2` | `41.36%` | `56.50%` | `42.60%` | `21.88 min` |
| `1.0` | `1` | `48.24%` | `51.38%` | `48.72%` | `32.22 min` |
| `1.0` | `2` | `64.32%` | `65.02%` | `63.81%` | `34.40 min` |
| `2.0` | `1` | `49.62%` | `50.90%` | `48.56%` | `32.48 min` |
| `2.0` | `2` | `50.08%` | `51.08%` | `50.64%` | `31.85 min` |
| `4.0` | `1` | `56.56%` | `57.36%` | `56.42%` | `31.33 min` |
| `4.0` | `2` | `40.44%` | `42.96%` | `40.17%` | `30.67 min` |
| `8.0` | `1` | `46.48%` | `47.08%` | `47.47%` | `29.52 min` |
| `8.0` | `2` | `55.02%` | `55.94%` | `54.13%` | `32.27 min` |

The runs grow from the `296`-parameter starter; completed C-sweep runs finish between `1,644` and `5,088` parameters. Their last recorded MCTS searches took `11.90–58.28s` and analyzed `15–104` actions. This exceeds the configured `8s` budget, so actual search time and work remain important alongside accuracy.

#### MCTS · UCB1 formula (`MCTS_UCB1_USE_SQRT`)

Both formula settings have completed two seeds. `False` leads with `50.52%` mean final validation accuracy and remains the selected setting.

| Setting | Completed runs | Mean final validation | Seed range | Mean final test |
|---|---:|---:|---:|---:|
| `False` **selected** | `2/2` | **`50.52%`** | `43.20–57.84%` | `50.59%` |
| `True` | `2/2` | `40.64%` | `38.38–42.90%` | `41.45%` |

![Training histories by UCB1 formula](/assets/experiments/010-pass-1-ucb1-use-sqrt-training-histories-grid.svg)

#### MCTS · Rollout value propagation (`MCTS_PROPAGATE_ROLLOUT_VALUE`)

Enabling rollout value propagation is currently best, with `57.39%` mean final validation accuracy across two seeds.

| Setting | Completed runs | Mean final validation | Seed range | Mean final test |
|---|---:|---:|---:|---:|
| `False` | `2/2` | `56.26%` | `50.94–61.58%` | `56.51%` |
| `True` **selected** | `2/2` | **`57.39%`** | `54.20–60.58%` | `57.00%` |

![Training histories by rollout value propagation](/assets/experiments/010-pass-1-propagate-rollout-value-training-histories-grid.svg)

#### MCTS · Rollout depth (`MCTS_ROLLOUT_DEPTH`)

Depth `0` leads this sweep at `51.80%` mean final validation accuracy; the default depth `2` is third.

| Depth | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| `0` **selected** | `2/2` | **`51.80%`** | `47.40–56.20%` | `51.47%` |
| `1` | `2/2` | `46.85%` | `43.78–49.92%` | `47.14%` |
| `2` (default) | `2/2` | `49.32%` | `46.00–52.64%` | `48.73%` |
| `3` | `2/2` | `46.90%` | `34.92–58.88%` | `47.38%` |

![Training histories by rollout depth](/assets/experiments/010-pass-1-rollout-depth-training-histories-grid.svg)

#### MCTS · Rollout training epochs (`MCTS_ROLLOUT_EPOCHS`)

Two rollout epochs perform best so far, with `51.60%` mean final validation accuracy.

| Epochs | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| `1` (default) | `2/2` | `47.89%` | `35.28–60.50%` | `48.18%` |
| `2` **selected** | `2/2` | **`51.60%`** | `48.18–55.02%` | `51.50%` |
| `4` | `2/2` | `34.96%` | `28.02–41.90%` | `34.80%` |

![Training histories by rollout epochs](/assets/experiments/010-pass-1-rollout-epochs-training-histories-grid.svg)

#### MCTS · Rollout learning rate (`MCTS_ROLLOUT_LR`)

The configured `0.0001` value is subject to the scheduler's `0.001` floor. Among the tested values, `0.01` currently leads with `48.10%` mean final validation accuracy.

| Configured rate | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| `0.0001` (default) | `2/2` | `37.39%` | `24.68–50.10%` | `37.06%` |
| `0.003` | `2/2` | `44.09%` | `38.76–49.42%` | `43.77%` |
| `0.01` **selected** | `2/2` | **`48.10%`** | `41.36–54.84%` | `48.56%` |

![Training histories by rollout learning rate](/assets/experiments/010-pass-1-rollout-lr-training-histories-grid.svg)

#### Simulation · Minimum algorithm iterations (`SIMULATION_MIN_ALGORITHM_ITERATION_RUNS`)

The value `1` leads this sweep with `46.46%` mean final validation accuracy and is selected for the remaining Pass 1 sweeps.

| Minimum iterations | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| `1` **selected** | `2/2` | **`46.46%`** | `45.68–47.24%` | `47.04%` |
| `3` (default) | `2/2` | `40.41%` | `38.62–42.20%` | `40.82%` |
| `5` | `2/2` | `42.28%` | `37.30–47.26%` | `42.14%` |

![Training histories by minimum simulation iterations](/assets/experiments/010-pass-1-simulation-min-algorithm-iteration-runs-training-histories-grid.svg)

#### Simulation · Time budget (`simulation_time`)

`16s` leads this sweep with `43.70%` mean final validation accuracy, slightly above `8s`; `32s` did not improve the result.

| Time budget | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| `8s` (default) | `2/2` | `41.97%` | `35.66–48.28%` | `42.42%` |
| `16s` **selected** | `2/2` | **`43.70%`** | `31.50–55.90%` | `44.08%` |
| `32s` | `2/2` | `40.28%` | `34.74–45.82%` | `39.99%` |

#### Simulation · Training-set percentage (`simulation_set_percent`)

This sweep is near completion. The `0.284%`, `5%`, `10%`, and `30%` settings have both seeds complete. Candidate `40%`, seed `1`, is complete at `64.26%` validation; seed `2` is at `120/200` epochs, so there is no selected value yet.

| Training-set share | Completed runs | Mean final validation | Seed range | Mean final test |
|---:|---:|---:|---:|---:|
| Current baseline (`0.284%`) | `2/2` | `39.29%` | `30.88–47.70%` | `40.41%` |
| `5%` | `2/2` | `49.02%` | `44.62–53.42%` | `49.37%` |
| `10%` | `2/2` | `57.16%` | `56.34–57.98%` | `56.96%` |
| `30%` | `2/2` | `61.79%` | `61.66–61.92%` | `61.44%` |
| `40%` **selected** | `2/2` | **`64.00%`** | `63.74–64.26%` | `64.39%` |

#### Remaining parameters

All nine Pass 1 sweeps are complete. Pass 2 has begun; its winners remain provisional until that pass is complete.

## Pass 2

Pass 2 retests each parameter from the Pass 1 winner configuration. Six sweeps are complete; minimum algorithm iterations is in progress. Results below report two-seed mean final validation accuracy, except active candidates.

### Meta-parameters

Every sweep starts with Pass 1 values `C=1.0`, UCB1 formula `False`, rollout propagation `True`, rollout depth `0`, rollout epochs `2`, rollout learning rate `0.01`, minimum iterations `1`, simulation time `16s`, and simulation training-set share `40%`. Only the parameter under test changes.

#### MCTS · UCB1 C

`C=1.0` leads at `64.11%` mean validation accuracy.

| C | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `0.5` | `2/2` | `61.93%` | `60.20–63.66%` | `62.03%` |
| `1.0` **selected** | `2/2` | **`64.11%`** | `63.24–64.98%` | `64.15%` |
| `2.0` | `2/2` | `60.42%` | `57.64–63.20%` | `60.13%` |
| `4.0` | `2/2` | `61.60%` | `61.58–61.62%` | `61.50%` |
| `8.0` | `2/2` | `62.16%` | `60.14–64.18%` | `61.64%` |

![Pass 2 UCB1 C training histories](/assets/experiments/010-pass-2-ucb1-c-training-histories-grid.svg)

#### MCTS · UCB1 formula

`False` remains ahead at `63.43%` mean validation.

| Square-root formula | Runs | Mean validation | Seed range | Mean test |
|---|---:|---:|---:|---:|
| `False` **selected** | `2/2` | **`63.43%`** | `63.04–63.82%` | `63.16%` |
| `True` | `2/2` | `61.92%` | `59.08–64.76%` | `62.16%` |

![Pass 2 UCB1 formula training histories](/assets/experiments/010-pass-2-ucb1-use-sqrt-training-histories-grid.svg)

#### MCTS · Rollout value propagation

The settings are effectively tied; `True` leads by `0.02` points.

| Propagation | Runs | Mean validation | Seed range | Mean test |
|---|---:|---:|---:|---:|
| `False` | `2/2` | `62.76%` | `61.30–64.22%` | `63.30%` |
| `True` **selected** | `2/2` | **`62.78%`** | `62.44–63.12%` | `62.77%` |

![Pass 2 rollout propagation training histories](/assets/experiments/010-pass-2-propagate-rollout-value-training-histories-grid.svg)

#### MCTS · Rollout depth

The sweep is complete: depth `1` is selected with the highest mean validation accuracy (`64.08%`).

| Depth | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `0` (Pass 1 value) | `2/2` | `60.49%` | `59.00–61.98%` | `60.24%` |
| `1` **selected** | `2/2` | **`64.08%`** | `63.10–65.06%` | `63.68%` |
| `2` | `2/2` | `60.26%` | `58.46–62.06%` | `60.18%` |
| `3` | `2/2` | `61.55%` | `59.88–63.22%` | `61.12%` |

![Pass 2 rollout depth training histories](/assets/experiments/010-pass-2-rollout-depth-training-histories-grid.svg)

#### MCTS · Rollout training epochs

This sweep is complete. Candidate `1` leads at `62.99%` mean validation, while candidate `4` finishes at `60.17%`.

| Rollout epochs | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `1` | `2/2` | **`62.99%`** | `61.32–64.66%` | `63.13%` |
| `2` (Pass 1 value) | `2/2` | `56.86%` | `54.88–58.84%` | `56.73%` |
| `4` | `2/2` | `60.17%` | `58.30–62.04%` | `60.21%` |

#### MCTS · Rollout learning rate

Candidate `0.0001` is selected by a narrow margin (`62.06%` mean validation versus `62.05%` for `0.01`). The Pass 1 choice `0.01` therefore does not win this retest.

| Configured rate | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `0.0001` **selected** | `2/2` | **`62.06%`** | `61.30–62.82%` | `62.46%` |
| `0.003` | `2/2` | `59.05%` | `54.86–63.24%` | `60.50%` |
| `0.01` (Pass 1 value) | `2/2` | `62.05%` | `61.70–62.40%` | `62.02%` |

#### Remaining Pass 2 parameters

#### Simulation · Minimum algorithm iterations

Candidate `3` is selected at `60.85%` mean validation accuracy, ahead of `1` and `5`.

| Minimum iterations | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `1` (Pass 1 value) | `2/2` | `58.72%` | `54.44–63.00%` | `58.84%` |
| `3` **selected** | `2/2` | **`60.85%`** | `57.26–64.44%` | `61.00%` |
| `5` | `2/2` | `55.79%` | `55.50–56.08%` | `55.87%` |

#### Simulation · Time budget

The sweep is complete. `32s` is selected with `60.73%` mean final validation, above `16s` (`58.58%`) and `8s` (`56.24%`).

| Time budget | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| `8s` | `2/2` | `56.24%` | `55.72–56.76%` | `56.62%` |
| `16s` (Pass 1 value) | `2/2` | **`58.58%`** | `56.50–60.66%` | `58.89%` |
| `32s` **selected** | `2/2` | **`60.73%`** | `57.06–64.40%` | `60.89%` |

#### Simulation · Training-set percentage

The sweep is complete. `10%` is selected at `59.16%` mean validation, ahead of `40%` (`58.40%`), `30%`, `5%`, and the baseline.

| Simulation training share | Runs | Mean validation | Seed range | Mean test |
|---:|---:|---:|---:|---:|
| Current baseline (`0.284%`) | `2/2` | `46.74%` | `39.20–54.28%` | `47.54%` |
| `5%` | `2/2` | `52.86%` | `52.34–53.38%` | `52.27%` |
| `10%` **selected** | `2/2` | **`59.16%`** | `58.24–60.08%` | `59.05%` |
| `30%` | `2/2` | `54.26%` | `51.68–56.84%` | `54.03%` |
| `40%` | `2/2` | `58.40%` | `56.84–59.96%` | `58.40%` |

## Conclusion

The final Pass 2 configuration is `C=1.0`, UCB1 formula `False`, rollout propagation `True`, rollout depth `1`, rollout epochs `1`, rollout learning rate `0.0001`, minimum iterations `3`, simulation time `32s`, and simulation training-set share `10%`. Each winner was chosen by the highest mean final validation accuracy within its sweep. These sequential one-parameter sweeps identify a tested configuration, not a joint search over all possible parameter combinations.

## Phase 3: 40-generation confirmation

Phase 3 repeats the full Pass 2 winning configuration three times on CIFAR-10. Each run uses the same tiny starter, `40` generations × `20` epochs (`800` epochs total), and seeds `1`, `2`, and `3`. The selected MCTS settings remain fixed throughout; Phase 3 is a longer-run confirmation, not another parameter sweep.

| Seed | Status | Final validation | Final test |
|---:|---|---:|---:|
| `1` | Not started | — | — |
| `2` | Not started | — | — |
| `3` | Not started | — | — |

Run the confirmation with `python -u experiments/train_cifar10_exp010_mcts_meta.py --phase3`. Results and histories are stored under `experiments/output/train_cifar10/runs/exp010_mcts_meta_g10_tiny/phase_3/`.
