[[Simulation Set]]

It keeps examples that would push the last-layer weights the most. The score is the L2 norm of the per-example last-layer gradient of cross-entropy. Short [[Scoring function]] training then spends its few epochs on high-impact points. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that gradient size predicts useful simulation signal.

Paper: Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021, 34, 20596-20607 (`paul2021dataDiet`). This page is the GraNd sampler. [[EL2N Simulation Set]] is the other Data Diet score in this repo.

Generating actions: `GrandSimulationSet.generate` in `growingnn/simulation/simulation_sets/grand.py`. `require_model` first. Grads from `per_example_last_layer_grads`. Default `selection_mode="highest"`. Per class, `class_balanced_top_scores` keeps about `K / C` points. `needs_refresh` is always `True`.

Pseudocode:

```text
require_model(model)
for each sample:
    GraNd_i = || d CE(model(x_i), y_i) / d last_layer ||_2
for each class:
    keep the top K/C scores
return Subset loaders
```

Executing actions: none.

Comparison with the original growingNN paper: the original sample ignored gradient size. GraNd keeps examples that would move the classifier most.

Known limitations: last-layer approximation, not full-network GraNd. Highest scores can keep noisy labels. Experiment 007 currently builds the set once on the untrained starter.
