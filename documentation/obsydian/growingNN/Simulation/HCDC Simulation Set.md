[[Simulation Set]]

It does not pick real training images. It learns a tiny synthetic set whose last-layer validation gradient matches real val data. Short [[Scoring function]] training then runs on those tensors. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that a condensed proxy preserves ranking of architecture moves.

Paper: Ding, Xu, Rabbani, Liu, Gravelle, Ranadive, Tuan, Huang, Calibrated Dataset Condensation for Faster Hyperparameter Search, arXiv:2405.17535 (`ding2024calibrated`). This GrowingNN version is a simplification.

Generating actions: `HcdcSimulationSet.generate` in `growingnn/simulation/simulation_sets/hcdc.py`. `require_model` first. `_condense` inits synthetic `x` from `protected_sampling_indices`. Default `steps=8`, `synthetic_lr=0.1`, `time_cap=20.0` seconds. Match loss is squared error between `mean_last_layer_grad` on val and `per_example_last_layer_grads` on synthetic, then Adam on `synthetic_x`. `needs_refresh` is `True` until the first successful build.

Pseudocode:

```text
require_model(model)
init synthetic x, y from a protected sample of size K
g_val = mean last-layer grad on real val
for up to 8 steps or 20 s:
    g_syn = mean last-layer grad on synthetic
    x := Adam(x, d||g_syn - g_val||^2 / dx)
return TensorDataset loaders
```

Executing actions: none. Loaders wrap a `TensorDataset`, not a `Subset`.

Comparison with the original growingNN paper: the original used real samples. HCDC synthesizes a tiny proxy set.

Known limitations: no implicit-function hypergradients over architecture variables. Spearman and Kendall are not the training loss. Condensation errors raise. Construction time can steal search budget. Experiment 007 currently builds the set once on the untrained starter.
