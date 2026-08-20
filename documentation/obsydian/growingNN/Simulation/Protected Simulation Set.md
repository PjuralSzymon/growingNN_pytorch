[[Simulation Set]]

It is the old GrowingNN control. It draws a class-balanced random subset. Every class keeps at least one example. Short [[Scoring function]] training uses this subset. Live training still uses the full loaders. Default `RunningConfig.simulation_set_generator` is this class. [[Model Drift Simulation Set]] also uses it as the default inner picker.

Paper: none. Same idea as the original growingNN simulation sample.

Generating actions: `ProtectedSimulationSet.generate` in `growingnn/simulation/simulation_sets/protected.py`. Train size is `min(size, len(train))`. Val size is `min(max(size // 4, 1), len(val))`. Indices from `protected_sampling_indices` in `commons.py`. About `n // C` random points per class. `needs_refresh` is `False`.

Pseudocode:

```text
K_train = min(K, |train|)
K_val = min(max(K // 4, 1), |val|)
for each class:
    take max(K_train // C, 1) random train indices
for each class:
    take max(K_val // C, 1) random val indices  (seed + 1)
return Subset loaders
```

Executing actions: none.

Comparison with the original growingNN paper: this is the old class-balanced random path (`sample_loaders` / `protected_sampling_indices`).

Known limitations: it does not look at the model. A tiny random set can still miss rare modes inside a class.
