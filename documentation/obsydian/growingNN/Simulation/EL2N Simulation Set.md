[[Simulation Set]]

It keeps examples whose current prediction is far from the one-hot label. The score is the L2 distance between softmax and the target. Short [[Scoring function]] training then sees currently hard points. It is cheaper than [[GraNd Simulation Set]] because it needs a forward pass only. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that prediction error predicts useful simulation signal.

Paper: Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet: Finding Important Examples Early in Training, NeurIPS 2021, 34, 20596-20607 (`paul2021dataDiet`).

Generating actions: `El2nSimulationSet.generate` in `growingnn/simulation/simulation_sets/el2n.py`. `require_model` first. `EL2N = ||softmax(logits) - one_hot(y)||_2`. Default `selection_mode="highest"`. Per class, `class_balanced_top_scores` keeps about `K / C` points. `needs_refresh` is always `True`. Middle-band selection is [[Moderate Difficulty Simulation Set]], not this class.

Pseudocode:

```text
require_model(model)
for each sample:
    EL2N_i = || softmax(model(x_i)) - one_hot(y_i) ||_2
for each class:
    keep the top K/C scores
return Subset loaders
```

Executing actions: none.

Comparison with the original growingNN paper: the original sample was random. EL2N keeps currently hard examples.

Known limitations: highest EL2N can keep noisy labels. Needs the current model. Experiment 007 currently builds the set once on the untrained starter.
