After training weights each generation, the algorithm may search for a better architecture. It samples a small data subset, explores legal moves from `growingnn/actions/registry.py`, grades rollouts, and returns one action to execute on the live model.

Code lives under `growingnn/simulation/`. `train_generations` in `growingnn/training/trainer.py` wires loaders, calls `config.simulation_alg.get_action` on a model copy, then runs `action.execute` on the live graph.

## Simulation modules

### Search

- [[MCTS]] — default Monte Carlo tree search (`montecarlo_alg.py`)

Alternatives on `RunningConfig.simulation_alg`: `greedy_alg.py`, `random_alg.py` (no separate vault pages).

### Grading

- [[Scoring function]] — weighted `SimulationScore` over acc, loss, time, and parameter count
- [[Simulation Set]] — how train/val loaders are sampled for rollouts
