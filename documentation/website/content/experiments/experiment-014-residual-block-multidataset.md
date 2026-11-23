# Experiment 014: residual-block actions across image datasets

## Context

Experiment 013 settled the initializer on short runs. This experiment runs the residual-block actions for longer (up to 300 epochs) from the same residual starter and on four datasets of increasing difficulty, from MNIST to Tiny ImageNet, to see whether the action helps once the network has time to grow. Each run with the actions has a matching run without them, same dataset and seed.

## Summary

- **Question:** do residual-block actions help when the starter already has a residual block?
- **Turning them off was better:** CIFAR-10 72.9% vs 63.9% test, CIFAR-100 34.4% vs 31.8%, MNIST about equal (Figure 3).
- **But no block action was applied** in any MNIST or CIFAR run, so this measures offering the action, not using it. Only Tiny ImageNet chose a duplicate (Figure 4).
- **Stopped early:** Tiny ImageNet finished one seed (22.0% test); the other runs were dropped because searches took about 175 s instead of 8 s.

| Setting | Value |
| --- | --- |
| Datasets, seeds | MNIST, CIFAR-10, CIFAR-100, Tiny ImageNet; seeds 1–3 |
| Starter | Conv16, BN, ReLU, one residual block, pool, Linear (2,970 parameters on CIFAR-10) |
| Schedule | Up to 30 generations × 10 epochs; stop at 90% validation |
| Search | Monte Carlo, 8 s, 128 examples (old defaults) |
| Duplicate initializer | Quasi-identity |

> [!CAPTION] Table 1. Setup. 19 of 24 planned runs finished; MNIST reached the 90% stop in 9–17 epochs.

## Results

![Training and validation accuracy for all runs](/assets/experiments/014-training-histories.png)

> [!CAPTION] Figure 1. Training (solid) and validation (dashed) accuracy; colour is the seed. The dotted line is the 90% stop.

![Parameter count by epoch](/assets/experiments/014-parameter-histories.png)

> [!CAPTION] Figure 2. Parameter count; solid traces with block actions, dashed without.

![Final test accuracy by seed](/assets/experiments/014-final-test-accuracy.png)

> [!CAPTION] Figure 3. Final test accuracy per finished seed; horizontal lines are arm means.

![Simulation scores of residual-block candidates](/assets/experiments/014-block-candidate-scores.png)

> [!CAPTION] Figure 4. Simulation scores of duplicate and remove candidates; bars are means. Only the two Tiny ImageNet duplicates were chosen.

## Conclusion

Offering the block actions made CIFAR-10 and CIFAR-100 runs worse on average, but the search never applied them there, so this is not an effect of the action. The likely reason is that the extra candidates change which other actions get sampled in a short, noisy search. The search did score the duplicate and chose it on Tiny ImageNet, so the action works mechanically. To learn whether it helps, it has to be tested directly on a fixed network, which is what Experiment 015 does. Another finding: the 8-second search actually took about 3 minutes on Tiny ImageNet, which led to the search-budget work in later experiments.

## Limitations

- Not finished: Tiny ImageNet has one enabled seed and no control.
- Three seeds; CIFAR-10 enabled seeds range 59.5–69.1%.
- No block action applied on MNIST or CIFAR, so nothing is learned about the action itself. Experiment 015 tests it directly on a checkpoint.

## Saved results

Script: `experiments/train_cifar10_exp014_residual_block_multidataset.py`. Runs: `experiments/output/train_cifar10/runs/exp012_residual_block_multidataset/`. Charts and snapshot: `generate_experiment_014_report.py`.
