[[Simulation Set]]

Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet, NeurIPS 2021 (`paul2021dataDiet`).

Generating actions: `El2nSimulationSet.generate` in `growingnn/simulation/simulation_sets/el2n.py`. `EL2N = ||softmax(logits) - one_hot(y)||_2` on the current model. Default selection is highest score, class balanced.

Executing actions: none.

Comparison with the original growingNN paper: the original sample was random. EL2N keeps currently hard examples.

Known limitations: highest EL2N can keep noisy labels. Middle-band selection is [[Moderate Difficulty Simulation Set]], not this class.
