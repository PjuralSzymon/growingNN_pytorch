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

**As of 11 November 2026, 09:58 UTC (10:58 Warsaw):** the experiment started on **10 November at 21:59 UTC (22:59 Warsaw)** and has run for about **12 hours**. **17 of 120 runs are complete**, with one more in progress; epoch-equivalent progress is **14.4%**.

At the observed pace, estimate **about 51 hours remaining**, with a projected finish around **13 November 2026, 14:15 Warsaw time**. This is a rough estimate and assumes continuous sequential runs. The current study uses the CIFAR-10 tiny starter; the earlier CIFAR-100 pilot is excluded.

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

The `False` formula setting is still running its two-seed comparison. Seed 1 completed at `43.20%` final validation; seed 2 is at `101/200` epochs with `52.44%` current validation and `54.98%` best validation so far. The `True` setting has not started, so there is no winner yet.

| UCB1 formula | Seed | Status | Epochs | Final/latest validation | Best validation | Final test | Runtime |
|---|---:|---|---:|---:|---:|---:|---:|
| `False` | `1` | Complete | `200/200` | `43.20%` | `48.08%` | `42.78%` | `31.67 min` |
| `False` | `2` | In progress | `101/200` | `52.44%` | `54.98%` | — | `9.10 min` so far |
| `True` | `1`, `2` | Not started | — | — | — | — | — |

#### Remaining parameters

Pending the UCB1 formula comparison.

## Pass 2

Pass 2 will repeat all nine parameter sweeps, starting from the pass 1 winners. It has not started.

### Meta-parameters

Pending pass 1.

## Conclusion

No MCTS meta-parameter winner has been established for this ten-generation, small-starter setup.
