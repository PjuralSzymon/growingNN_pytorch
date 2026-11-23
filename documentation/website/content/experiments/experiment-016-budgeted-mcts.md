# Experiment 016: Budgeted search on CIFAR-10

## Context

Experiments 010 and 014 showed that the Monte Carlo search often runs far past its time budget (up to 20 times longer). This experiment tries a simpler budgeted search: each step scores one new candidate, the search stops hard at the deadline, and the final action is the candidate with the best average score. It starts from the same tiny 296-parameter network as Experiment 010.

## Summary

- **Growth:** the budgeted search grew the 296-parameter starter to 2,432 parameters and reached **53.9% validation / 54.1% test** (Figure 1).
- **Small models:** it mostly added linear layers and dropout; no widening was selected.
- **Kept for Monte Carlo 2:** selecting by mean score, counting each score once, and no random rollout actions.
- **Dropped:** the hard deadline, because it could skip candidates.

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10, seed 1 |
| Starter | Conv 3→8, ReLU, pool, Linear 8→10 (296 parameters) |
| Schedule | 10 generations × 20 epochs |
| Search | Budgeted: one new candidate per iteration, hard 32 s stop, root actions only, pick the highest mean score |

> [!CAPTION] Table 1. Setup. Each search scored 5–7 candidates in 34–41 s.

## Training and growth

![Experiment 016 accuracy with the action after each generation, and parameter count](/assets/experiments/016-training-and-parameters.png)

> [!CAPTION] Figure 1. Training and validation accuracy (top) and parameters on a log scale (bottom). Gray lines are generation boundaries (20 epochs), labelled with the action applied there.

The network grew in small steps: the first convolution with BatchNorm gave the largest jump, and later actions were mostly linear layers and dropout that added few parameters. The model ended with only 2,432 parameters, too small to go much beyond 54%.

## Conclusion

The budgeted search keeps to its time limit, but it grew the network too little. Three of its ideas were kept and later became part of Monte Carlo 2: choose the action with the best average score, count each score once, and skip random follow-up actions. The hard deadline was dropped, because stopping early means some candidates are never scored.

## Limitations

- One seed, and no matching run of the original Monte Carlo search.
- The search could stop before scoring every candidate.
- Its code is removed; a `source_snapshot.zip` is saved with the run.

[Experiment 017](experiment-017-family-search.md) compares this run with the family-balanced search.

## Saved results

Run folder (not tracked by git): `experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/`. Chart: `generate_experiment_016_017_charts.py`.
