# Experiment 010: MCTS meta-parameter tuning

## Context

This experiment looks for the best way to run the Monte Carlo tree search (MCTS) that chooses each change to the growing network.

## Part 1: MCTS variants

We started by comparing the existing Monte Carlo search with two new variants, Stabilized and Extended. The comparison ran once, in a sibling repository, on CIFAR-10 from a tiny starter network, in two rounds with two seeds each: Phase 1 with short training and Phase 2 with longer training and a stronger training recipe. The variants brought no benefit: the original search had the best result in both phases, so we kept it. The results are shown below.

![Mean test accuracy of the three search algorithms in both phases](/assets/experiments/010-mcts-variants.png)

> [!CAPTION] Figure 1. Mean test accuracy (two seeds) of the three algorithms in Phase 1 (short training) and Phase 2 (longer training). The pre-study used a different code version and schedule from Part 2, so these numbers are not comparable with the tuning results.

| Algorithm | How it picks the next change |
| --- | --- |
| **Existing Monte Carlo** | Builds a search tree of changes, trains each changed copy briefly, and uses UCB1 to decide which branches to explore. |
| **Stabilized** | Trains each changed copy and an unchanged copy with the same seed and time, and scores the difference. The best changes are re-tested with new seeds and applied only if they still beat the unchanged copy. |
| **Extended** | Tests short sequences of changes, each scored against an unchanged copy, and focuses on promising sequences. The best first changes are re-tested before one is applied. |

> [!CAPTION] Table 1. The three algorithms.

## Part 2: tuning the Monte Carlo search

We then tuned the nine meta-parameters of the existing Monte Carlo search, listed below. Tuning ran in two passes. In the first pass we started from the default settings and tested every value of one meta-parameter, kept the best one, and moved on to the next meta-parameter with that winner fixed. In the second pass we repeated all nine sweeps starting from the Pass 1 winners, to check whether the choices still held. All runs use CIFAR-10, the 296-parameter starter (Conv 3→8, pool, Linear), 10 generations × 20 epochs and seeds 1 and 2; the best value is the one with the highest mean final validation accuracy.

| Meta-parameter | What it controls | Values tested |
| --- | --- | --- |
| UCB1 C | How much the search explores less-visited changes | 0.5, 1, **2**, 4, 8 |
| UCB1 square root | Which UCB1 formula is used | **no**, yes |
| Propagate rollout value | Whether a node passes its latest score up the tree | **no**, yes |
| Rollout depth | Random follow-up changes added before scoring | 0, 1, **2**, 3 |
| Rollout epochs | Training epochs per scored copy | **1**, 2, 4 |
| Rollout learning rate | Learning rate when training a scored copy | **1e-4**, 3e-3, 1e-2 |
| Minimum iterations | Search steps made even after the time runs out | 1, **3**, 5 |
| Search time | Time budget per search | **8 s**, 16 s, 32 s |
| Simulation set | Share of the training data used to score copies | **0.3%** (128 images), 5%, 10%, 30%, 40% |

> [!CAPTION] Table 2. The nine meta-parameters; bold is the default.

### UCB1 exploration constant (`MCTS_UCB1_C`)

The first meta-parameter checked was the UCB1 exploration constant C. In Pass 1 the best value was 1 (56.3%), and in Pass 2 it was also 1 (64.1%).

![UCB1 C in both passes](/assets/experiments/010-sweep-ucb1-c.png)

> [!CAPTION] Figure 2. Final validation accuracy for each value of UCB1 C, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 0.5 | 46.6% (41.4–51.9) | 61.9% (60.2–63.7) |
| 1 | **56.3%** (48.2–64.3) | **64.1%** (63.2–65.0) |
| 2 (default) | 49.8% (49.6–50.1) | 60.4% (57.6–63.2) |
| 4 | 48.5% (40.4–56.6) | 61.6% (61.6–61.6) |
| 8 | 50.8% (46.5–55.0) | 62.2% (60.1–64.2) |

> [!CAPTION] Table 3. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

The training histories show how large the seed noise is: with C = 1 the two seeds end 16 pp apart.

![Pass 1 training histories, UCB1 C](/assets/experiments/010-pass-1-ucb1-c-training-histories-grid.svg)

> [!CAPTION] Figure 3. Pass 1 training (solid) and validation (dashed) accuracy for each value; blue seed 1, orange seed 2. Gray lines are generation boundaries (20 epochs).

### UCB1 formula (`MCTS_UCB1_USE_SQRT`)

Next was the UCB1 formula. The default formula without the square root won both passes (50.5% and 63.4%).

![UCB1 formula in both passes](/assets/experiments/010-sweep-ucb1-sqrt.png)

> [!CAPTION] Figure 4. Final validation accuracy for each value of UCB1 formula, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| no (default) | **50.5%** (43.2–57.8) | **63.4%** (63.0–63.8) |
| yes | 40.6% (38.4–42.9) | 61.9% (59.1–64.8) |

> [!CAPTION] Table 4. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Rollout value propagation (`MCTS_PROPAGATE_ROLLOUT_VALUE`)

Turning rollout value propagation on won both passes, but by only 1.1 pp and 0.02 pp: effectively a tie.

![rollout propagation in both passes](/assets/experiments/010-sweep-propagate-rollout.png)

> [!CAPTION] Figure 5. Final validation accuracy for each value of rollout propagation, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| no (default) | 56.3% (50.9–61.6) | 62.8% (61.3–64.2) |
| yes | **57.4%** (54.2–60.6) | **62.8%** (62.4–63.1) |

> [!CAPTION] Table 5. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Rollout depth (`MCTS_ROLLOUT_DEPTH`)

Depth 0 was best in Pass 1 (51.8%) and depth 1 in Pass 2 (64.1%). The default depth 2 never won.

![rollout depth in both passes](/assets/experiments/010-sweep-rollout-depth.png)

> [!CAPTION] Figure 6. Final validation accuracy for each value of rollout depth, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 0 | **51.8%** (47.4–56.2) | 60.5% (59.0–62.0) |
| 1 | 46.9% (43.8–49.9) | **64.1%** (63.1–65.1) |
| 2 (default) | 49.3% (46.0–52.6) | 60.3% (58.5–62.1) |
| 3 | 46.9% (34.9–58.9) | 61.5% (59.9–63.2) |

> [!CAPTION] Table 6. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Rollout training epochs (`MCTS_ROLLOUT_EPOCHS`)

2 rollout epochs were best in Pass 1 (51.6%) and 1 epoch in Pass 2 (63.0%).

![rollout epochs in both passes](/assets/experiments/010-sweep-rollout-epochs.png)

> [!CAPTION] Figure 7. Final validation accuracy for each value of rollout epochs, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 1 (default) | 47.9% (35.3–60.5) | **63.0%** (61.3–64.7) |
| 2 | **51.6%** (48.2–55.0) | 56.9% (54.9–58.8) |
| 4 | 35.0% (28.0–41.9) | 60.2% (58.3–62.0) |

> [!CAPTION] Table 7. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

With 4 rollout epochs both seeds drop sharply after generation 2 and stay below about 35% for most of the run.

![Pass 1 training histories, Rollout epochs](/assets/experiments/010-pass-1-rollout-epochs-training-histories-grid.svg)

> [!CAPTION] Figure 8. Pass 1 training (solid) and validation (dashed) accuracy for each value; blue seed 1, orange seed 2. Gray lines are generation boundaries (20 epochs).

### Rollout learning rate (`MCTS_ROLLOUT_LR`)

1e-2 was best in Pass 1 (48.1%). In Pass 2, 1e-4 and 1e-2 tied (62.06% and 62.05%) and 1e-4 was kept.

![rollout learning rate in both passes](/assets/experiments/010-sweep-rollout-lr.png)

> [!CAPTION] Figure 9. Final validation accuracy for each value of rollout learning rate, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 1e-4 (default) | 37.4% (24.7–50.1) | **62.1%** (61.3–62.8) |
| 3e-3 | 44.1% (38.8–49.4) | 59.0% (54.9–63.2) |
| 1e-2 | **48.1%** (41.4–54.8) | 62.0% (61.7–62.4) |

> [!CAPTION] Table 8. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Minimum search iterations (`SIMULATION_MIN_ALGORITHM_ITERATION_RUNS`)

1 was best in Pass 1 (46.5%) and the default 3 in Pass 2 (60.9%).

![minimum iterations in both passes](/assets/experiments/010-sweep-min-iterations.png)

> [!CAPTION] Figure 10. Final validation accuracy for each value of minimum iterations, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 1 | **46.5%** (45.7–47.2) | 58.7% (54.4–63.0) |
| 3 (default) | 40.4% (38.6–42.2) | **60.8%** (57.3–64.4) |
| 5 | 42.3% (37.3–47.3) | 55.8% (55.5–56.1) |

> [!CAPTION] Table 9. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Search time (`simulation_time`)

16 s was best in Pass 1 (43.7%) and 32 s in Pass 2 (60.7%).

![search time in both passes](/assets/experiments/010-sweep-search-time.png)

> [!CAPTION] Figure 11. Final validation accuracy for each value of search time, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 8 s (default) | 42.0% (35.7–48.3) | 56.2% (55.7–56.8) |
| 16 s | **43.7%** (31.5–55.9) | 58.6% (56.5–60.7) |
| 32 s | 40.3% (34.7–45.8) | **60.7%** (57.1–64.4) |

> [!CAPTION] Table 10. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

### Simulation set size (`simulation_set_percent`)

40% was best in Pass 1 (64.0%) and 10% in Pass 2 (59.2%). The default 128 images (0.3%) was the worst in both passes, 10–25 pp below the best: the largest effect in the whole study.

![simulation set size in both passes](/assets/experiments/010-sweep-simulation-set.png)

> [!CAPTION] Figure 12. Final validation accuracy for each value of simulation set size, Pass 1 (left) and Pass 2 (right). Small dots are the two seeds, the large marker their mean; orange is the selected value.

| Value | Pass 1 | Pass 2 |
| --- | ---: | ---: |
| 0.3% (default) | 39.3% (30.9–47.7) | 46.7% (39.2–54.3) |
| 5% | 49.0% (44.6–53.4) | 52.9% (52.3–53.4) |
| 10% | 57.2% (56.3–58.0) | **59.2%** (58.2–60.1) |
| 30% | 61.8% (61.7–61.9) | 54.3% (51.7–56.8) |
| 40% | **64.0%** (63.7–64.3) | 58.4% (56.8–60.0) |

> [!CAPTION] Table 11. Mean final validation accuracy (the two seeds in brackets); bold is the selected value.

With 128 images (0.3%) or 5%, the runs stay near 35% for several generations and the two seeds split apart. From 10% on, both seeds climb from the second generation and stay close together.

![Pass 1 training histories, Sim. set](/assets/experiments/010-pass-1-simulation-set-percent-training-histories-grid.svg)

> [!CAPTION] Figure 13. Pass 1 training (solid) and validation (dashed) accuracy for each value; blue seed 1, orange seed 2. Gray lines are generation boundaries (20 epochs).

## Limitations

- Two seeds per value, while seeds differ by up to 25 pp. The selected configuration is a reasonable choice, not a proven optimum.
- One meta-parameter at a time; interactions were not tested.
- A planned 40-generation confirmation (Phase 3) was not run.

## Saved results

Script: `experiments/train_cifar10_exp010_mcts_meta.py`. Data snapshot with every run: `documentation/website/data/experiments/experiment-010-mcts-meta.json`. The runs are in `exp010_mcts_meta_g10_tiny/` of the `growingNN_pytorch-3` checkout; pass that folder to `generate_experiment_010_charts.py`. Charts: `generate_experiment_010_overview.py` and `generate_experiment_010_charts.py`.

## Summary

In Part 1 the existing Monte Carlo search beat both new variants, so it was kept. In Part 2 its nine meta-parameters were tuned in two passes; the figure shows every tested value and the table gives the final set of parameters, which is the default search in the experiments that follow.

![Final validation accuracy for every candidate value in both passes](/assets/experiments/010-sweep-overview.png)

> [!CAPTION] Figure 14. Final validation accuracy of every tested value, Pass 1 (top) and Pass 2 (bottom). The two small dots are the two seeds, the large marker their mean; orange marks the selected value. `*` marks the default.

| Meta-parameter | Default | Pass 1 winner | **Final (Pass 2 winner)** |
| --- | --- | --- | --- |
| UCB1 C | 2 | 1 | **1** |
| UCB1 square root | no | no | **no** |
| Propagate rollout value | no | yes | **yes** |
| Rollout depth | 2 | 0 | **1** |
| Rollout epochs | 1 | 2 | **1** |
| Rollout learning rate | 1e-4 | 1e-2 | **1e-4** |
| Minimum iterations | 3 | 1 | **3** |
| Search time | 8 s | 16 s | **32 s** |
| Simulation set | 0.3% | 40% | **10%** |

> [!CAPTION] Table 12. Selected value per meta-parameter.

- **The tuned search reaches about 59–64% validation**, against about 50% with the defaults.
- **The simulation set size matters most:** the default 128 images was far too small.
- **The other meta-parameters matter less than the random seed:** five of the nine Pass 1 winners changed in Pass 2.
