[[Simulation]]

The simulation set is a small train/val pair used only while search grades architecture actions. Live training still uses the full loaders. `train_generations` in `growingnn/training/trainer.py` asks `RunningConfig.simulation_set.generate` to build `sim_train_loader` and `sim_val_loader`. If `needs_refresh` is true before `get_action`, the set is built again from the live model.

The default generator is `ProtectedSimulationSet` in `growingnn/simulation/simulation_sets/base.py`. It keeps the old class-balanced random sample. Size is `RunningConfig.simulation_set_size`. Seed is `RunningConfig.simulation_set_seed`.

Other generators live under `growingnn/simulation/simulation_sets/` and implement the same `SimulationSet` interface: [[Moderate Difficulty Simulation Set]], [[Model Drift Simulation Set]], [[GRAD-MATCH Simulation Set]], [[HCDC Simulation Set]], [[k-Center Simulation Set]], [[GraNd Simulation Set]], [[EL2N Simulation Set]], [[CRAIG Simulation Set]]. Shared helpers are in `commons.py`.

Known limitations: model-aware generators need an `nn.Linear` classifier. HCDC v1 matches last-layer validation gradients, not full hypergradients from Ding et al. 2024.
