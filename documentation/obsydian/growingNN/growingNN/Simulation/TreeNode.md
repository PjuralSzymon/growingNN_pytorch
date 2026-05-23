A `TreeNode` is one node in the MCTS tree used by [[MCTS]]. It holds a model copy `M`, parent link, the `action` that led here, train data `X_train` / `Y_train`, epoch count, and a `simulation_score` object. It tracks `value` (sum of rollout scores), `visit_counter`, and `childNodes`.

---

### `expand`

What. Creates one child for every legal action at this node.

How. Calls `Action.generate_all_actions(self.M)`. For each action:

1. `M_copy = self.M.deepcopy()`
2. `action.execute(M_copy)`
3. One training step: `gradient_descent(..., 1, LearningRateScheduler(CONSTANT, 0.0001, 0.8), ...)`
4. Append a new `TreeNode` with that action and `M_copy`.

Why. First-level children represent all moves the search can compare at the root after a minimal weight update.

---

### `rollout`

What. Plays out a random action sequence and returns one [[Scoring function]] score.

How. Starts from `M.deepcopy()`. Repeats up to `DEEPTH` times (constant `2`):

1. List actions with `Action.generate_all_actions(M_copy)`.
2. Stop if the list is empty or only contains `Empty_action`.
3. Pick `random.choice(all_action_seq)`, execute, run one `gradient_descent` step as in `expand`.
4. Remove actions that `can_be_influenced` by the chosen action from the local list.
5. Decrease depth counter.

Then call `simulation_score.scoreFun(M_copy, self.epochs, self.X_train, self.Y_train)`.

Why. Rollout gives a cheap estimate of how good a branch is before UCB1 invests more visits.

---

### `get_best_child`

Uses UCB1 with exploration constant `UCB1_CONTS = 2`:

`score = node.value + UCB1_CONTS * log(parent.visit_counter) / visit_counter`

Unvisited children (`visit_counter == 0`) get score `inf` so each is tried. Returns the child with the highest score.

---

### `kill`

Walks children, clears `childNodes`, drops references to `M`, data, and parent. Used after [[MCTS]] `get_action` finishes so large model copies do not linger.

---

### Known limitations

`protected_divide` guards UCB1 when visit counts grow huge. Rollout depth `2` is fixed; it does not adapt to model size. Each expand trains every child for one epoch, which is costly when many actions exist.
