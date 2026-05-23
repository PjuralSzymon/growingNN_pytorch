Monte Carlo Tree Search picks one architecture action after each [[Training loop]] stage in a generation. It is stage 2 in [[General]]. Each candidate move comes from `Action.generate_all_actions` (see [[Actions]]). Each rollout is graded by the [[Scoring function]] on a [[Simulation Set]].

We use a modified MCTS, not a full game tree. Two changes matter most.

1. A rollout ends after a fixed number of actions (`DEEPTH = 2`), not when the model reaches a final state.
2. Search is time limited, but it must visit every root action at least once. The loop keeps running while there is time left or while `rollouts <= size_of_changes`, where `size_of_changes` is the count of legal actions at the root.

---

### Entry point: `get_action`

Inputs: model `M`, time budget `max_time_for_dec`, training epoch count `epochs`, train tensors `X_train`, `Y_train`, and `simulation_score`.

Steps:

1. Build `size_of_changes = len(Action.generate_all_actions(M))`. If zero, return no action.
2. Create root [[TreeNode]] with the current model and no action yet.
3. Set `deadline = time.time() + max_time_for_dec`.
4. Repeat `simulate(root)` while there is time left or while `rollouts <= size_of_changes`. Search stops only when the deadline has passed and every root action has been rolled out at least once.
5. Pick `root.get_best_child().action` with UCB1 among first-level children only.
6. Call `root.kill()` and `clear_reshepers_cache()` to free memory.
7. Return `best_action`, max tree depth seen, and rollout count.

The chosen action is always one legal move from the current model, not a deep sequence.

---

### Search step: `simulate`

Classic MCTS phases, adapted to our tree:

1. At a leaf with `visit_counter == 0`: run `rollout()` once, store score, count one rollout.
2. At a leaf already visited: call `expand()` to create one child per legal action at that model state.
3. Pick a child with `get_best_child()` (UCB1, constant `UCB1_CONTS = 2`).
4. Recurse into that child, add the returned score to `node.value`, increment `visit_counter`.

---

### Comparison with the original growingNN paper

DOI 10.1007/978-3-031-63749-0_25 describes search over architectures with simulation. This tree search matches that idea. Fixed rollout depth and a time floor for root actions are implementation choices from the original codebase, not spelled out in the chapter.

---

### Known limitations

Search depth at the root is guaranteed; deeper levels depend on remaining time. Rollouts use one short `gradient_descent` step per action with a constant learning rate (`0.0001`), not the full [[Learning Rate Scheduler]] from training. Child selection uses average accumulated rollout score as `node.value`, not a strict mean of wins.
