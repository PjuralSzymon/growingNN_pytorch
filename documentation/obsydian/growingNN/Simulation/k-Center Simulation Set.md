[[Simulation Set]]

It covers the current model's feature space. Each new point is the one farthest from the points already chosen. Short [[Scoring function]] training then sees a diverse subset, not a cluster of similar images. It is used as `RunningConfig.simulation_set_generator` when the hypothesis is that coverage of embeddings makes architecture scores more stable.

Paper: Sener and Savarese, Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR 2018 (`sener2018coreSet`).

Generating actions: `KCenterSimulationSet.generate` in `growingnn/simulation/simulation_sets/kcenter.py`. `require_model` first. Embeddings are `extract_embeddings` (input to the last `nn.Linear`). Per class, `_kcenter_indices` uses `torch.cdist`. First point is nearest the class mean. Then greedy farthest-first until `class_quota`. `needs_refresh` is always `True`.

Pseudocode:

```text
require_model(model)
embed = embeddings(model, train)  # before last Linear
for each class:
    S = {point nearest class mean embed}
    while |S| < K/C:
        add argmax_i min_{j in S} ||embed_i - embed_j||
return Subset loaders
```

Executing actions: none. `indices_to_loaders` builds the sim pair for [[Scoring function]].

Comparison with the original growingNN paper: the original sample was random inside each class. k-Center spreads points across feature space.

Known limitations: needs the current model. Cost grows with train-set size times selected count per class. A single Linear classifier has embeddings equal to the raw input. Experiment 007 currently builds the set once on the untrained starter.
