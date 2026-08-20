[[Simulation Set]]

Paul, Ganguli, Dziugaite, Deep Learning on a Data Diet, NeurIPS 2021 (`paul2021dataDiet`). This file is the GraNd sampler.

Generating actions: `GrandSimulationSet.generate` in `growingnn/simulation/simulation_sets/grand.py`. Score is the L2 norm of the last-layer per-example gradient. Default selection is highest score, class balanced.

Executing actions: none.

Comparison with the original growingNN paper: the original sample ignored gradient size. GraNd keeps examples that would move the classifier most.

Known limitations: last-layer approximation, not full-network GraNd.
