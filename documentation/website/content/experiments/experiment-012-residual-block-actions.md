# Experiment 012: residual block actions on small image models

## Context

The search grows a network by adding single layers. Successful hand-designed networks such as ResNet are built from repeated residual blocks, so a new pair of actions was added: duplicate an existing residual block, or remove one. This experiment is the first check of whether the search uses them and whether runs that may use them end better than runs that may not.

## Summary

- **Question:** does adding the residual-block grow/shrink actions help the search?
- **The search never chose them:** 0 of 228 selected actions were residual-block actions (Figure 3), so this experiment cannot say whether they help.
- **Arm differences are seed noise:** the "off" arm was higher on MNIST and CIFAR-10 and lower on CIFAR-100, with one MNIST seed stuck at 63% (Figure 2).
- **Missing data:** candidate scores were not saved, which led to the score logging used from Experiment 013 on.

| Setting | Value |
| --- | --- |
| Datasets, seeds | MNIST, CIFAR-10, CIFAR-100; seeds 1 and 2 per arm |
| Starter | Conv →8, ReLU, pool, Linear (152–1,016 parameters) |
| Schedule | 20 generations × 10 epochs |
| Search | Monte Carlo, 8 s, 1 epoch, 128 simulation examples (old defaults) |
| Arms | Residual-block actions on vs off; all other actions on in both |

> [!CAPTION] Table 1. Setup.

## Results

![Training and validation accuracy per seed and arm](/assets/experiments/012-training-histories.png)

> [!CAPTION] Figure 1. Training (solid) and validation (dashed) accuracy over 200 epochs; blue is the arm with block actions on, orange off.

![Final test accuracy by seed and arm](/assets/experiments/012-final-test-accuracy.png)

> [!CAPTION] Figure 2. Final test accuracy per seed; horizontal marks are the arm means.

![Selected action counts per dataset and arm](/assets/experiments/012-action-counts.png)

> [!CAPTION] Figure 3. Selected actions summed over both seeds (38 per group). No residual-block action was ever selected.

## Conclusion

The question could not be answered: the starter network has no residual block, so the action had nothing to work on at the start, and the search never chose it later. The differences between the arms therefore come from seeds and from small changes in which other actions were sampled, not from the new action. Experiments 013 and 014 start from a network that already contains a residual block, and from Experiment 013 on every candidate's score is saved.

## Limitations

- Two seeds per cell; no block action was applied, so there is no evidence either way about the action.
- Only small image starters were tested.

## Saved results

Script: `experiments/train_cifar10_exp012_residual_block_actions.py`. Runs: `experiments/output/train_cifar10/runs/exp010_residual_block_actions/` (folder name predates the renumbering). Snapshot: `documentation/website/data/experiments/experiment-012-residual-block-actions.json`. Charts: `generate_experiment_012_block_actions_report_data.py`.
