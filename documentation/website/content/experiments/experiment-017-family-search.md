# Experiment 017: Growing a small network on CIFAR-10

## Context

The budgeted search of Experiment 016 grew the tiny starter to only 2,432 parameters and 54%. One suspected reason was that it spent its few evaluations on many variants of the same action type and rarely tried widening. The family-balanced search scores one candidate of each action type in turn, and adds widening across residual branches. It uses the same starter, schedule and data as Experiment 016.

## Summary

- **Growth works:** the search grew a 296-parameter starter to 80,887 parameters and reached **69.5% validation / 69.8% test** in 200 epochs (Figure 1).
- **It beat the budgeted search of Experiment 016 by 15.6 pp** (53.9% → 69.5%, Figure 2), but on one seed and with several search changes at once, so the cause is not isolated.
- **One action made most of the difference:** a 64→64 convolution with BatchNorm and ReLU after epoch 80, followed by +20.8 pp over the next generation (Figure 1).
- **The search picked it by a small margin** (39.2% vs 37.3% simulation score, Figure 3), so the choice was partly luck.

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10, seed 1 |
| Starter | Conv 3→8, ReLU, global pool, Linear 8→10 (296 parameters) |
| Schedule | 10 generations × 20 epochs |
| Search | Monte Carlo, root actions only (depth 0), ~32 s per search, 4,500 / 1,120 simulation images |
| New in this search | Candidates spread across action types; widening across residual branches |

> [!CAPTION] Table 1. Setup. Experiment 016 used the same starter and schedule with the budgeted search.

## Training and growth

![Experiment 017 training and validation accuracy with the action after each generation, and parameter count](/assets/experiments/017-training-and-parameters.png)

> [!CAPTION] Figure 1. Training and validation accuracy per epoch (top) and parameter count on a log scale (bottom). Gray lines are generation boundaries (20 epochs); the label at each line is the action applied there. The jump after epoch 80 follows the added convolution with BatchNorm and ReLU.

![Validation accuracy and parameter growth of experiments 016 and 017](/assets/experiments/016-017-validation-and-growth.png)

> [!CAPTION] Figure 2. Validation accuracy (top) and parameters on a log scale (bottom) for Experiment 016 (budgeted search) and 017 (family-balanced search), one seed each.

The jump after epoch 80 did not happen at insertion: the new convolution starts as an identity, so validation stayed at 33.7% and rose to 46.5% after one more epoch and to 54.6% after the generation. The 20-epoch gains after each action are not causal estimates; the model kept training and there is no unchanged-model control.

## Why the epoch-80 action was chosen

![Candidate scores in the search after epoch 80](/assets/experiments/017-action-scores-gen-3.png)

> [!CAPTION] Figure 3. Simulation-validation accuracy (one training epoch on the simulation subset) of each candidate in the search after epoch 80. The star marks the selected action; `n` is the number of evaluations.

![Candidate scores of every search in Experiment 017](/assets/experiments/017-action-scores-overview.png)

> [!CAPTION] Figure 4. Candidate scores of all nine searches, one row per generation, with fixed positions per action type. A blank position means no candidate of that type was scored. Generations 5–8 scored only 5 of 21 generated actions because the time budget ran out.

Each search scored 5 candidates once and spent the sixth evaluation on the leader; the action with the highest mean score won. Later searches offered similar convolutions in other places, but other actions scored higher.

## Conclusion

This run showed that the tiny starter can reach about 70% if the search widens it early and then adds a convolution with BatchNorm. It also showed how fragile the choice was: the decisive action won its search by under 2 pp, inside the score noise. That decision became the test case of Experiment 018, which replays it many times to make the search pick it reliably.

## What was kept

The family-balanced search has since been removed. Its fan-out widening is now part of the library's `WidenConv` action, and its decision checkpoints are what Experiment 018 replays. Spreading candidates across action types was dropped: Monte Carlo 2 scores every root action at least once.

## Limitations

- One seed. A seed-2 run was interrupted before its first search.
- The search could stop before scoring every candidate (Figure 4).
- The search order and the action set changed together, so the gain over Experiment 016 cannot be assigned to either.

## Saved results

Run folder (not tracked by git): `experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1/`, with `run.json` and the board. The 016 baseline is in `exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/`. All candidate scores: [017-action-score-details.csv](/assets/experiments/017-action-score-details.csv). Charts: `documentation/website/scripts/generate_experiment_016_017_charts.py` and `generate_experiment_017_action_scores.py`.
