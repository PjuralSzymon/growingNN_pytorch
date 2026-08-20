[[Simulation Set]]

Middle-quantile sampling per class by current-model cross-entropy. Related motivation is Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet, NeurIPS 2021 (`paul2021dataDiet`). That paper is not the exact sampler.

Generating actions: `ModerateDifficultySimulationSet.generate` in `growingnn/simulation/simulation_sets/moderate_difficulty.py`. Sort each class easy to hard, keep `[0.25, 0.75)`, then `evenly_spaced_select`.

Executing actions: none. This only builds loaders for [[Scoring function]].

Comparison with the original growingNN paper: the paper used a small simulation sample. This replaces uniform class sampling with a middle-difficulty band.

Known limitations: needs the current model. Extremes are dropped on purpose.
