Efficiency terms in `growingnn/simulation/score_functions/score_efficiency.py`. Used by [[Scoring function]] when `weight_time` or `weight_countW` is greater than 0.

- `score_time` runs one quiet `gradient_descent` (same epoch count and loaders as learning terms), measures wall time, then returns `1 / (TIME_EFFICIENCY_WEIGHT * elapsed + 1)`. `TIME_EFFICIENCY_WEIGHT` is `1.0` in `growingnn/core/config.py`. Faster rollouts score closer to 1.

- `score_count_weights` counts parameters with `GraphStructureQuery.get_amount_of_parameters` and returns `1 / (param_count * WEIGHT_COUNT_WEIGHT + 1)`. `WEIGHT_COUNT_WEIGHT` is `1e-6` in config. Smaller models score higher. No training step.

Comparison with the original growingNN paper: the paper combined accuracy with a size or cost term. Here size is parameter count; time is an extra optional term.