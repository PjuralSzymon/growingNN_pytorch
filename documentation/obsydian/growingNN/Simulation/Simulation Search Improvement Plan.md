[[Simulation]]

This page is the plan for replacing or comparing Monte Carlo tree search with clearer search methods. The goal is a time-limited simulation that always returns a usable action, and that gets better as more time is spent. The live trainer still applies one action per generation through `config.simulation_alg.get_action` in `growingnn/training/trainer.py`. Candidate methods are listed on [[Candidate Simulation Algorithms]]. Shared expand and score steps are on [[Simulation Search Common Steps]].

## Business summary

GrowingNN trains weights, then sometimes searches for an architecture change. Search must pick one legal action from `generate_all_actions` in `growingnn/actions/registry.py`. There is no true end of game. A rollout is only a short sequence of mutations plus a grade from `SimulationScore`.

Current default search is [[MCTS]]. Past work showed MCTS beats random and beats depth-1 greedy on average. Still, many runs here show high variance. Live grades of root children are often very close. Sometimes the chosen action later hurts accuracy on the real model. That means the search signal is weak, noisy, or misaligned with the real training outcome.

The needed product behavior is simple. Under a short time budget, return at least one scored action at depth 1. If more time remains, keep searching so the chosen action improves. Prefer clear bookkeeping of which actions were tried and what score they got. Prefer using action type and target layer to prune the set before expensive grades. Keep short selective lookahead, because depth-1 greedy cannot see multi-step plans such as add then remove. Do not rely on long uncontrolled random futures to invent a good first move.

## Mathematical view

At generation `t` the live network is a state `s_t` in a discrete space `S` of valid FX graphs. The legal action set is `A(s) subset A`. Each action `a in A(s)` maps to a child state `T(s, a)`.

There is no terminal reward. A grade is a noisy oracle

`r(s') = SimulationScore(s') + noise`

after a short simulated train on the simulation set from [[Simulation Set]]. The search problem for one generation is

`a* = argmax_{a in A(s_t)} V(s_t, a)`

where `V` is the estimated value of applying `a` now. Value may include depth-1 grade only, or a short lookahead over future actions. The wall-clock budget is `B` seconds. An algorithm is anytime if after every finished evaluation it can return a current best `a_hat`, and if more budget usually improves `a_hat` or its confidence.

This is finite-horizon search on a large branching tree with expensive noisy edge evaluations. It is closer to budgeted combinatorial optimization and multi-armed bandits than to perfect-information games with a win/loss terminal state.

## Tree search view

The search tree has root `s_t`. An edge is one architecture action. A node is a mutated graph plus optional short train state. Branching factor is `|A(s)|` and grows fast with layer count. Depth is truncated because training never ends. In GrowingNN only the first edge from the root is executed on the live model. Deeper edges exist only to estimate whether the first edge is promising.

So the tree is an estimation device for root edges, not a plan of many live mutations. Good algorithms should spend much of the budget on ranking root children under noise. They should also spend leftover budget on selective deepen so multi-step futures can still raise or lower a root edge. Random expand-all rollouts are the part to replace, not lookahead itself.

Useful structure inside `A(s)`:

- action family (add residual conv, delete layer, change neuron count, dropout, ...)
- target layer ids and local graph neighborhood
- cheap filters before full `SimulationScore` (size jump, invalid residual shapes already blocked by generators)

[[MCTS]] today expands all children, runs short GD on expand and rollout (`MCTS_ROLLOUT_*` in `growingnn/core/config.py`), then grades with another scoring GD. Selection uses UCB1. Default UCB uses cumulative value without the classic mean and square-root form unless `MCTS_UCB1_USE_SQRT` is on. Backup can inflate values when `MCTS_PROPAGATE_ROLLOUT_VALUE` is false. Those details can make close noisy scores look interchangeable and can favor unstable deep rollouts.

## Problems we have

1. Wrong live actions. Search can pick a root action that later collapses accuracy. There is no accept or revert in `train_generations` after the action runs.
2. Weak discrimination. Candidate grades are often nearly tied, so UCB exploration noise can dominate.
3. Random deep rollouts. Future random adds and deletes can hide which first action was bad.
4. Expensive and uneven evaluation. Expand, rollout, and scoring each may train. Greedy search scores differently from MCTS, so comparisons are unfair.
5. Unused structure. Action type and layer identity do not guide pruning. `can_be_infulenced` is unused.
6. Exponential branching. More layers mean more legal actions. Full expand-all trees do not scale.
7. Signal mismatch. Simulation set, short epochs, and live recovery LR are not the same as the next real generation.

## Rules the next algorithm must follow

1. Time limited. Stop when `simulation_scheduler.simulation_time` is used, but always finish a defined base case.
2. Base case first. At minimum, evaluate depth-1 candidates until each root action has a grade, or until a declared top-k cover is complete if full cover is impossible under the budget.
3. Anytime. After the base case, every extra evaluation should only improve the current best choice or its confidence.
4. Return one root action. Same contract as `get_action(...) -> (action, max_depth, rollouts)`.
5. Clear scores. Store per-action grades that a human can read on the experiment board.
6. Prefer simplicity. Reuse known graph or bandit methods before inventing a new search.
7. Use structure when cheap. Group or prune by action family and layer before full scoring when possible.
8. Keep scoring comparable. All algorithms under test should call the same `SimulationScore` path from [[Scoring function]].
9. Stay modular. New methods plug in through `RunningConfig.simulation_alg` like `montecarlo_alg`, `greedy_alg`, and `random_alg`.

## Experiment plan shape

Build a common harness that swaps only `simulation_alg` and shared search knobs (budget, max depth, beam width, top-k). Keep model, data, seeds, and [[Scoring function]] fixed. Measure for each algorithm:

- chosen action family and layer target
- simulated score of the winner and of the runner-up gap
- real train and val accuracy change over a fixed post-action window
- fraction of harmful actions (accuracy drop beyond a threshold)
- wall time and number of scored states

Start with the top ten on [[Candidate Simulation Algorithms]]. Wave 1: Sequential Halving, Beam Search, and Sequential Halving then Beam. Wave 2: UGapE and SHOT, then Progressive Widening if branching is too large. Keep current MCTS and greedy as anchors. Pick the best two for a larger MNIST or CIFAR grid.

## Comparison with the original growingNN paper

The paper used Monte Carlo style simulation to choose architecture moves under limited simulation. This plan keeps that product goal. It questions whether classic MCTS with random rollouts is the right estimator when only the first action is applied and grades are noisy.

## Known limitations of this plan

This page does not yet name a winner. Algorithm fit is reasoned from search theory and from current code behavior, not from a finished head-to-head in this repo. Accept or revert after a bad live action is a separate safety layer and is not a substitute for better search.
