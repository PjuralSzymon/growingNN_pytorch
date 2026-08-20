[[Simulation Set]]

It picks a small weighted subset whose last-layer gradient points the same way as the full training set. Short [[Scoring function]] SGD then sees a similar update direction. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that matching the current SGD direction makes architecture scores more useful.

Paper: Killamsetty, S, Ramakrishnan, De, Iyer, GRAD-MATCH: Gradient Matching based Data Subset Selection for Efficient Deep Model Training, ICML 2021, PMLR 139, 5464-5474 (`pmlr-v139-killamsetty21a`).

Generating actions: `GradMatchSimulationSet.generate` in `growingnn/simulation/simulation_sets/grad_match.py`. `require_model` first. Reference is `mean_last_layer_grad` on the train loader. Candidates come from `ground_set_indices` (stratified cap `size * 4`). Per class, `omp_select` matches last-layer grads from `collect_last_layer_grads`. `needs_refresh` is always `True`.

Pseudocode:

```text
require_model(model)
g_full = mean last-layer grad over train
candidates = stratified cap, size * 4
for each class:
    residual = g_full
    for k in 1 .. K/C:
        pick unused point with largest |g_i · residual|
        re-fit weights so weighted grads ~ g_full
        residual = g_full - weighted sum
sample with WeightedRandomSampler(weights)
```

Executing actions: none. `indices_to_loaders` can attach OMP weights via `WeightedRandomSampler`.

Comparison with the original growingNN paper: the original sample was random inside each class. GRAD-MATCH keeps the SGD direction of the current model.

Known limitations: last-layer only. The candidate pool is a cap, not every train example. Needs the current model. Experiment 007 currently builds the set once on the untrained starter.
