[[Simulation Set]]

Sener and Savarese, Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR 2018 (`sener2018coreSet`).

Generating actions: `KCenterSimulationSet.generate` in `growingnn/simulation/simulation_sets/kcenter.py`. Embeddings are the current GrowingNN features before the last `nn.Linear`. Per class, greedy k-Center with `torch.cdist`.

Executing actions: none.

Comparison with the original growingNN paper: the original sample was random inside each class. k-Center spreads points across feature space.

Known limitations: needs the current model. Cost grows with train-set size times selected count per class.
