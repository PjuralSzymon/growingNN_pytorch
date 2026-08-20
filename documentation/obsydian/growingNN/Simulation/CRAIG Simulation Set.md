[[Simulation Set]]

It covers last-layer gradient space with a small weighted coreset. Similar gradients are represented by one point. That point gets a weight equal to how many gradients it stands for. Short [[Scoring function]] training then samples with those weights. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that covering gradient space is better than picking the largest gradients.

Paper: Mirzasoleiman, Bilmes, Leskovec, Coresets for Data-efficient Training of Machine Learning Models, ICML 2020, PMLR 119, 6950-6960 (`pmlr-v119-mirzasoleiman20a`).

Generating actions: `CraigSimulationSet.generate` in `growingnn/simulation/simulation_sets/craig.py`. `require_model` first. Last-layer grads from `collect_last_layer_grads` on a stratified cap from `ground_set_indices`. Per class, `craig_select` does greedy facility location. Already chosen rows get `CraigSimulationSet.ALREADY_SELECTED_IMPROVEMENT` (`float("-inf")`). `needs_refresh` is always `True`.

Pseudocode:

```text
require_model(model)
g_i = last-layer grad of sample i
for each class:
    S = {argmin_j sum_i ||g_i - g_j||}
    while |S| < K/C:
        add the point that most reduces sum_i min_{j in S} ||g_i - g_j||
    weight_j = how many points assign to j
sample with WeightedRandomSampler(weights)
```

Executing actions: none. `WeightedRandomSampler` uses those weights.

Comparison with the original growingNN paper: the original sample treated every point equally. CRAIG covers gradient space with a weighted coreset.

Known limitations: last-layer only. The candidate pool is a stratified cap. Experiment 007 currently builds the set once on the untrained starter.
