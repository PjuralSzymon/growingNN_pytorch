# Experiment 015: duplicate-action design

## Context

Experiments 012–014 could not tell whether the residual-block duplicate helps, because the search almost never applied it. This experiment separates the two questions. Part A applies the action directly to a fixed trained network and compares it with no change, so it measures the action itself. Part B runs the normal growing search with four design changes added one by one, to see whether the search now picks it.

## Summary

- **Question:** do four design changes make the residual-block duplicate more useful, and does the search choose it more often?
- **The redesigned duplicate helps when applied:** inserting the copy after the block's ReLU gives **+4.2 pp** over no change after one generation, against +1.9 pp for the original (Figure 2).
- **The search still rarely chooses it** (Figures 7–8): its ~2 pp advantage after one epoch is lost in ~12 pp of score noise among ~150 candidates.
- **Final accuracies of the growing runs are seed noise:** seeds differ by up to 19 pp, more than any design step (Figure 5).
- **The high level (~72%) comes from the Experiment 014 setup**, not from this action (Figure 1).

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Starter | Conv 3→16, BN, ReLU, residual block, pool, Linear (2,970 parameters) |
| Part A | One action on a fixed trained checkpoint, then one generation (10 epochs), 2 seeds |
| Part B | Growing runs, 30 generations × 10 epochs, seeds 1 and 2, old search defaults (8 s, 128 examples) |

> [!CAPTION] Table 1. Setup. Part A isolates the action; Part B tests it inside growth.

| Step | Change (cumulative) |
| --- | --- |
| off | Residual-block actions disabled |
| current | Original duplicate (copy before the ReLU) and hard remove |
| +side | Blocks with side inputs can be duplicated |
| +after_relu | The copy goes after the block's ReLU |
| +scale | Residual scale ε; ε = 1 won, so it repeats `+after_relu` |
| +merge | Shrink merges two identical neighbours instead of deleting |

> [!CAPTION] Table 2. The design ladder.

![Validation accuracy of Experiment 010, 014 and 015 setups](/assets/experiments/015-context.png)

> [!CAPTION] Figure 1. Validation accuracy at the end of each setup's budget, mean of seeds. The larger starter with BatchNorm and a residual block, not the search, explains the higher level.

## Part A: the action on a fixed checkpoint

![Validation gain over no change after one generation, per duplicate variant](/assets/experiments/015-paired-ladder.png)

> [!CAPTION] Figure 2. Validation after one generation minus no change, from the same checkpoint. Bars: mean of 2 seeds; markers: seeds. Left: plain checkpoint. Right: a checkpoint where the block has a side input.

![Validation accuracy over time after each action](/assets/experiments/015-paired-trajectory.png)

> [!CAPTION] Figure 3. Validation at the checkpoint, right after the action, after 1 epoch and after 10 epochs. The duplicate dips 4.4 pp, recovers after one epoch and pulls ahead later, after the point where the search scores it. Right: remove and merge both cost 9–13 pp.

![Effect of the residual scale on the duplicate](/assets/experiments/015-scale-sweep.png)

> [!CAPTION] Figure 4. Gain over no change at five residual scales ε. ε = 1 is best after one generation (+4.0 pp); ε = 0, an exact no-op copy, gives +2.8 pp.

## Part B: growing runs

![Final test accuracy for each step](/assets/experiments/015-final-test.png)

> [!CAPTION] Figure 5. Final test accuracy after 300 epochs; bars are means, circles seed 1, squares seed 2. The best step (`+after_relu`) never applied a block action.

![Validation accuracy histories for each step](/assets/experiments/015-val-histories.png)

> [!CAPTION] Figure 6. Validation per epoch; solid seed 1, dashed seed 2, gray the `off` mean. Green lines mark applied duplicates, red a remove: removing the only block in `+side` seed 1 left it 15 pp behind for the rest of the run.

![How often the duplicate is offered, ranked in the top 5, and chosen](/assets/experiments/015-duplicate-selection.png)

> [!CAPTION] Figure 7. Over 58 searches per step: how often a duplicate was offered, ranked in the top 5, and chosen. Side-input support doubled availability but not selection.

![Duplicate score compared with no change and with the winner](/assets/experiments/015-score-gaps.png)

> [!CAPTION] Figure 8. Best duplicate's score minus the no-change score (left) and minus the winner's score (right), per search. The gray band is ±1 SD of score noise. The duplicate scores like no change; the winner is the luckiest of ~150 candidates.

![Executed action types per step](/assets/experiments/015-action-mix.png)

> [!CAPTION] Figure 9. Applied actions by type, both seeds (58 per step). Residual and sequential convolutions dominate; block actions are 0–2 per step.

## Conclusion

The action is good: placed after the ReLU, the duplicate adds about 4 pp over one generation on a fixed network. The problem is the search. It scores each candidate after a single short training run, and the noise in those scores (about 12 pp) is much larger than the duplicate's early advantage (about 2 pp), so among roughly 150 candidates some other action almost always scores higher by luck. Improving the action further will not help until the scoring is more reliable, which is the starting point for Experiment 018.

## What changed afterwards

The duplicate now always goes after the ReLU with plain quasi-identity weights, works on blocks with side inputs (without copying the side input), and shrink merges identical neighbours or removes a block but never the last one. The scoring problem led to Monte Carlo 2 (Experiment 018), which also selects by mean score instead of the exploration bonus.

## Limitations

- Two seeds; seed spread exceeds the effects. Only 6 block actions ran in 10 growing runs; no merge ran.
- Part A uses small checkpoints at 50–60% validation; gains may differ on larger models.

## Saved results

Script: `experiments/train_cifar10_exp015_duplicate_action_design.py`. Runs: `experiments/output/train_cifar10/runs/exp014_duplicate_action_design/`. Charts and snapshot: `generate_experiment_015_report.py`.
