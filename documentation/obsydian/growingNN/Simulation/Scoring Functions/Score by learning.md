Learning terms in `growingnn/simulation/score_functions/score_by_learning.py`. Used by [[Scoring function]] when `weight_acc` or `weight_loss` is greater than 0.

Both call `gradient_descent` for `simulation_scheduler.simulation_epochs` on `RunningConfig.sim_train_loader` and `sim_val_loader` (built by `RunningConfig.simulation_set.generate` in `train_generations`). LR comes from `RunningConfig.lr_scheduler`.

- `score_acc` returns the last `val_acc` from the history (higher is better).
- `score_loss` returns `min(1 / (max(val_loss, 1e-8) + 1), 1.0)` so lower loss scores higher.

Comparison with the original growingNN paper: the paper used training-set metrics after simulation steps. This code uses validation metrics (`val_acc`, `val_loss`).