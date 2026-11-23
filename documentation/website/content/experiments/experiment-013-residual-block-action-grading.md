# Experiment 013: residual-block duplicate initialization

## Context

Experiment 012 never applied the residual-block action, partly because its starter had no block. This experiment starts from a network that already contains one, so the duplicate is available from the first search. It also asks how the new block should start. A copy of the existing weights changes what the network computes; an identity-style start (quasi-identity, or a residual branch set to zero) leaves the output almost unchanged, so the network keeps what it has learned and the new block trains from there.

## Summary

- **Question:** how should a duplicated residual block be initialized so the search rates it well?
- **Identity-style initializers score much higher than copied weights** (CIFAR-10: 0.45 vs 0.29 mean candidate score, Figure 3), so quasi-identity is now the only initializer.
- **Final accuracy does not separate the initializers:** differences are smaller than the seed spread (Figure 2).
- **Duplicates were applied only 3 times in 18 runs;** each raised training accuracy over the next generation but not validation (Figure 4).

| Setting | Value |
| --- | --- |
| Datasets, seeds | CIFAR-10 and CIFAR-100, seeds 1–3 |
| Starter | Conv16, BN, ReLU, one identity residual block, pool, Linear |
| Schedule | 10 generations × 5 epochs |
| Search | Monte Carlo, 4 s, 256 examples, old defaults |
| Arms | Copy, quasi-identity and residual-zero duplicates; block actions off (shared control) |

> [!CAPTION] Table 1. Setup: 30 runs (18 with block actions, 12 controls). A follow-up protocol for MNIST and CIFAR-10 with the Experiment 010 search was planned on this page but never run; Experiment 015 continued the question.

## Results

![Training and validation histories by initializer](/assets/experiments/013-training-histories.png)

> [!CAPTION] Figure 1. Training (solid) and validation (dashed) accuracy per initializer and dataset; colour is the seed, gray the controls. Dotted lines are generation boundaries (5 epochs).

![Final test accuracy by initializer and dataset](/assets/experiments/013-initialization-test-comparison.png)

> [!CAPTION] Figure 2. Final test accuracy per seed; black marks are means. Quasi-identity leads CIFAR-10 by 0.6 pp, residual-zero leads CIFAR-100; both within seed noise.

![Residual-block candidate scores by initializer](/assets/experiments/013-block-grades-by-initialization.png)

> [!CAPTION] Figure 3. Simulation scores of residual-block candidates by initializer (triangles: means, orange: medians). Copied weights score clearly lower on both datasets. Scores include up to two random rollout actions.

![Training history around each applied duplicate](/assets/experiments/013-block-duplicate-impact.png)

> [!CAPTION] Figure 4. The three applied duplicates: the shaded generation follows the action. Training accuracy rose by 1.3–3.8 pp, validation changed by −4.0 to −0.1 pp, and each added 2,336 parameters.

![Accuracy change after each selected action type](/assets/experiments/013-action-effects.png)

> [!CAPTION] Figure 5. Change in training and validation accuracy over the generation after each selected action, by action type. Dots are single actions, bars means. Descriptive only: training continues during the window.

## Conclusion

The initializer changes how the search rates the duplicate, not how the runs end. Copied weights make the duplicate look clearly worse in the short simulation, so it is almost never picked; both identity-style starts score about equally. Quasi-identity was chosen as the only initializer because it scored as well as residual-zero and was already tested in the project. Whether the duplicate itself improves a network is still open here, because it was applied only three times.

## Limitations

- Three seeds; short 50-epoch runs; old search settings.
- Initializer arms are not paired checkpoints, so their trajectories diverge after the first different choice.

## Saved results

Script: `experiments/train_cifar10_exp013_residual_block_action_grading.py`. Runs: `experiments/output/train_cifar10/runs/exp011_residual_block_action_grading/`. Charts: `generate_experiment_013_grading_charts.py` and `generate_experiment_013_grading_history_charts.py`.
