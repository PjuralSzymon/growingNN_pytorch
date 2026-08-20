[[Simulation]]

The simulation set is a small train/val pair used only while search grades architecture actions. Live training still uses the full loaders. Callers pass `sim_train_loader` and `sim_val_loader` into `train_generations` in `growingnn/training/trainer.py`. If those are missing, the trainer builds them once with `RunningConfig.simulation_set_generator.generate(train_loader, val_loader, simulation_set_size)`.

The default generator is `ProtectedSimulationSet` in `growingnn/simulation/simulation_sets/protected.py`. It keeps the old class-balanced random sample. Size is `RunningConfig.simulation_set_size`. Experiment runs seed with `seed_all` in `growingnn/utils/seed.py` before generate. The experiment runner samples from the unaugmented train loader, then passes the ready sim loaders into `train_generations`.

Other generators live under `growingnn/simulation/simulation_sets/` and implement the same `SimulationSet` interface: [[Moderate Difficulty Simulation Set]], [[Model Drift Simulation Set]], [[GRAD-MATCH Simulation Set]], [[HCDC Simulation Set]], [[k-Center Simulation Set]], [[GraNd Simulation Set]], [[EL2N Simulation Set]], [[CRAIG Simulation Set]]. Shared helpers are in `commons.py`.

Known limitations: model-aware generators need an `nn.Linear` classifier and raise if the model is missing. They do not fall back to `ProtectedSimulationSet`. HCDC v1 matches last-layer validation gradients, not full hypergradients from Ding et al. 2024.
