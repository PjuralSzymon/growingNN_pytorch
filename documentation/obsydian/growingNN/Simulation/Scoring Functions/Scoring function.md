`SimulationScore` in `growingnn/simulation/score_functions/simulation_score.py` ranks a candidate model after a simulated mutation. Greedy search (`greedy_alg.py`) and [[MCTS]] call `simulation_score.score(model, ctx)` on each rollout.

The user picks non-negative weights. The final grade is a weighted mean of active terms only (weights with value 0 are skipped so it's enought to set 0 in config and some score funciton won't be run). Keys: `weight_acc`, `weight_loss`, `weight_time`, `weight_countW`. Default weights in code: acc `1.0`, loss `0.0`, time `0.0`, countW `0.5`.

Sub-scores live in [[Score by learning]] and [[Score by effitiency]]. All of them read `RunningConfig` for epochs, LR, `criterion`, and simulation loaders (`set_simulation_loaders` in `train_generations` from [[Simulation Set]] sampling).

Known limitations: each enabled term runs on its own `copy.deepcopy(model)` and may run full `gradient_descent` separately. There is no shared cache between terms.
