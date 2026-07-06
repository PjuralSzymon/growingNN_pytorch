[[Simulation]]

The simulation set is a small subset of the training data used only during simulation. In early work we tried other ways to build the set, including PCA-based reduction. Those methods scored worse in practice, so random sampling stayed the default.

`train_generations` calls `sample_loaders` to build `RunningConfig.sim_train_loader` and `sim_val_loader` before the generation loop. Rollouts in `montecarlo_alg.py`, `greedy_alg.py`, and `score_by_learning.py` read those loaders.

Size is set by `RunningConfig.simulation_set_size` in `growingnn/core/config.py`.
