[[Simulation Set]]

Ding et al., Calibrated Dataset Condensation for Faster Hyperparameter Search, arXiv:2405.17535 (`ding2024calibrated`).

Generating actions: `HcdcSimulationSet.generate` in `growingnn/simulation/simulation_sets/hcdc.py`. Builds synthetic tensors. Matches last-layer validation gradients of the current model between real val data and the synthetic set.

Executing actions: none. Loaders wrap a `TensorDataset`, not a `Subset`.

Comparison with the original growingNN paper: the original used real samples. HCDC synthesizes a tiny proxy set.

Known limitations: this is a simplification of the paper. It does not use implicit-function hypergradients over architecture variables. Spearman and Kendall are not the training loss. Condensation errors raise.
