[[Simulation Set]]

Proposed method. No direct paper. Rebuilds an inner [[Simulation Set]] when embeddings on a fixed anchor set drift.

Generating actions: `ModelDriftSimulationSet` in `growingnn/simulation/simulation_sets/model_drift.py`. Default inner selector is `ProtectedSimulationSet`. Anchor size 256. Drift is mean cosine distance. Threshold 0.1.

Executing actions: none. `needs_refresh` is true when embeddings on the anchor set drift. The trainer does not call generate again; the caller builds the set before `train_generations`.

Comparison with the original growingNN paper: the original sample was built once. This refreshes when the live model representation moves.

Known limitations: a single linear classifier has embeddings equal to the raw input, so weight-only drift may stay below the threshold.
