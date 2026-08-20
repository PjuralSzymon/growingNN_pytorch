[[Simulation Set]]

Killamsetty et al., GRAD-MATCH, ICML 2021, PMLR 139, 5464-5474 (`pmlr-v139-killamsetty21a`).

Generating actions: `GradMatchSimulationSet.generate` in `growingnn/simulation/simulation_sets/grad_match.py`. Last-layer gradients. OMP matches a weighted subset gradient to the full-data gradient, per class.

Executing actions: none. Optional `WeightedRandomSampler` uses the OMP weights.

Comparison with the original growingNN paper: the original sample was random. GRAD-MATCH keeps the SGD direction of the current model.

Known limitations: last-layer only. The candidate pool is a stratified cap, not every train example.
