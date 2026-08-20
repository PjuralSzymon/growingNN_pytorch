After training weights each generation, the algorithm may search for a better architecture. It samples a small data subset, explores legal moves from `growingnn/actions/registry.py`, grades rollouts, and returns one action to execute on the live model.

Code lives under `growingnn/simulation/`. `train_generations` in `growingnn/training/trainer.py` wires loaders, calls `config.simulation_alg.get_action` on a model copy, then runs `action.execute` on the live graph.

## Simulation modules

### Search

- [[MCTS]] — default Monte Carlo tree search (`montecarlo_alg.py`)
- [[Simulation Search Improvement Plan]] — why MCTS is not enough and what rules the next search must follow
- [[Simulation Search Common Steps]] — shared words and expand / score / depth steps for every candidate
- [[Candidate Simulation Algorithms]] — keep-set methods after Experiment 005 (lookahead and hybrids)

Alternatives on `RunningConfig.simulation_alg` / Exp 005 `hp["simulation_alg"]`: `greedy_alg.py`, `random_alg.py`, and the keep-set modules listed on [[Candidate Simulation Algorithms]].

### Grading

- [[Scoring function]] — weighted `SimulationScore` over acc, loss, time, and parameter count
- [[Simulation Set]] — how train/val loaders are sampled for rollouts
