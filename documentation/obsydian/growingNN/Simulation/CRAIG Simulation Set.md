[[Simulation Set]]

Mirzasoleiman, Bilmes, Leskovec, Coresets for Data-efficient Training of Machine Learning Models, ICML 2020, PMLR 119, 6950-6960 (`pmlr-v119-mirzasoleiman20a`).

Generating actions: `CraigSimulationSet.generate` in `growingnn/simulation/simulation_sets/craig.py`. Last-layer gradient vectors. Greedy facility location, then one weight per selected point equal to how many gradients it represents.

Executing actions: none. `WeightedRandomSampler` uses those weights.

Comparison with the original growingNN paper: the original sample treated every point equally. CRAIG covers gradient space with a weighted coreset.

Known limitations: last-layer only. The candidate pool is a stratified cap.
