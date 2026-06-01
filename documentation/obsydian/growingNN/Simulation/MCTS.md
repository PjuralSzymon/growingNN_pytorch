We use Monte Carlo Tree Search with our own changes. It runs in the simulation stage of each generation. Simulations are based on all possiblie generated [[Actions]] in given generation . Each rollout is scored by the [[Scoring function]] on the [[Simulation Set]].

There are two main modifications.

1. The end of the "game" is not a win or loss state. A rollout stops after N actions.
2. Each search is time limited, but it must still analyze every action at the first depth. Search continues while there is time left or while `rollouts <= size_of_changes`, where `size_of_changes` is how many legal actions exist at the root. So every action that can be chosen from the current model is rolled out at least once; deeper levels are explored only if time remains.

## Algorithm

The simulation works as follows.

1. For a given model state in the current generation:
   1. Generate all possible actions.
   2. Grade moves with the [[Scoring function]] inside rollouts 
   3. Choose which branch to explore next with MCTS and UCB1 (`MCTS_UCB1_C`, `MCTS_UCB1_USE_SQRT` in [[Config]]). `_simulate` backprop uses `MCTS_PROPAGATE_ROLLOUT_VALUE`: default `False` returns accumulated `node.value`; `True` returns only the latest rollout score.
2. Repeat selection, expansion, and rollout until the time rule in point 2 above is satisfied.
3. Return one best action at the first level of the tree only (`root.get_best_child().action`). The search does not return a long action sequence.

Entry point in the original code is `get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score)`. It builds a root tree node, calls `simulate(root)` in a loop, then frees the tree with `kill()` and `clear_reshepers_cache()`.

Inside `simulate`, a leaf visited for the first time runs `rollout()`. A leaf visited again calls `expand()` and creates one child per legal action. Each child gets one short training step before UCB1 compares them.

![[Pasted image 20260523093226.png]]
