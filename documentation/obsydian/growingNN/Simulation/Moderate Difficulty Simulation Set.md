[[Simulation Set]]

It picks a small train subset that is neither very easy nor very hard for the current GrowingNN model. Easy points give almost no learning signal. The hardest points can be outliers or noisy labels. Short [[Scoring function]] training uses this subset, not the full dataset. Live training still uses the full loaders.

Paper: none for this exact sampler. Related only: Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet, NeurIPS 2021, 34, 20596-20607 (`paul2021dataDiet`).

Generating actions: `ModerateDifficultySimulationSet.generate` in `growingnn/simulation/simulation_sets/moderate_difficulty.py`. `require_model` first. Score is per-example `F.cross_entropy` with `reduction="none"`. Per class, sort easy to hard, keep quantile band `[0.25, 0.75)`, then `evenly_spaced_select` with `class_quota`. Default quantiles are constructor fields `lower_quantile=0.25` and `upper_quantile=0.75`. `needs_refresh` is always `True`.

Pseudocode:

```text
require_model(model)
score_i = CrossEntropy(model(x_i), y_i)
for each class:
    sort easy -> hard
    keep middle quantile [0.25 n, 0.75 n)
    take evenly spaced K / C indices
return Subset loaders
```

Executing actions: none. `indices_to_loaders` builds the sim train/val pair for [[Scoring function]].

Comparison with the original growingNN paper: the paper used a small class-balanced random sample. This keeps only the middle difficulty band of the current model.

Known limitations: needs the current model and an `nn.Linear` path that can score the set. Extremes are dropped on purpose. Experiment 007 currently builds the set once on the untrained starter.
