[[Simulation Set]]

It does not choose which examples to keep. It decides when an inner generator should build the set again. GrowingNN changes weights and architecture over generations, so a subset chosen on an old model can go stale. It is used as `RunningConfig.simulation_set_generator` when the caller wants a refresh policy around another picker such as [[Protected Simulation Set]] or [[k-Center Simulation Set]].

Paper: none. GrowingNN-specific. No direct reference.

Generating actions: `ModelDriftSimulationSet.generate` in `growingnn/simulation/simulation_sets/model_drift.py`. Default inner selector is `ProtectedSimulationSet`. Anchor size `256`. Drift is `mean_cosine_distance` on embeddings from `extract_embeddings` (activations before the last `nn.Linear`). Threshold `0.1`. `needs_refresh` is `True` when the cache is empty or mean cosine distance is at least `0.1`.

Pseudocode:

```text
require_model(model)
embed = embeddings(model, fixed 256-example anchor)
if cache empty or mean_cosine_distance(embed, last_embed) >= 0.1:
    sim_set = selector.generate(train, val, K, seed, model)
    last_embed = embed
else:
    keep cached sim_set
return sim_set
```

Executing actions: none. The current `train_generations` in `growingnn/training/trainer.py` does not call generate again. The caller must build the set, or call `generate` when `needs_refresh` is true.

Comparison with the original growingNN paper: the original sample was built once. This method is meant to rebuild when the live representation moves.

Known limitations: a single linear classifier has embeddings equal to the raw input, so weight-only drift may stay below `0.1`. Experiment 007 builds the set once on the untrained starter, so drift refresh is not exercised there.
