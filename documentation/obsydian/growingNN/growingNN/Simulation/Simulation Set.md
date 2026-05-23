The simulation set is a small subset of the training data used only during [[MCTS]]. [[Scoring function]] accuracy is measured on this set, not on the full train split.

Why a subset. Each rollout copies the model, applies actions, runs short training, and scores the result. Doing that on the full dataset would be too slow when many actions and rollouts run per generation.

What we use. A random sample from the training set. In early work we tried other ways to build the set, including PCA-based reduction. Those methods scored worse in practice, so random sampling stayed the default.

Where it is passed. `get_action` in [[MCTS]] receives `X_train` and `Y_train`; in simulation these tensors are the simulation set (or a slice of train chosen upstream).

---

### Comparison with the original growingNN paper

The paper argues for cheap simulation during search. Sub-sampling train data follows that goal. The choice of random sampling over PCA is a result from our own experiments, not a claim in the chapter.

---

### Known limitations

A small random set adds noise. A bad sample can rank the wrong action highly for one generation. The set is usually fixed for one search call, not reshuffled every rollout.
