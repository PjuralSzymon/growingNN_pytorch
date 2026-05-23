The scoring function judges how good a model is after a simulated action sequence. [[MCTS]] calls it at the end of each [[TreeNode]] `rollout` through `simulation_score.scoreFun(M, epochs, X_train, Y_train)`.

What it does. It turns a trained model copy and data into one number. Higher is better for UCB1 in [[MCTS]].

Why. Search must compare many architecture moves without running full training. A fast score on the [[Simulation Set]] approximates quality.

---

### Score terms

The user defines a weighted sum of sub-scores. The original setup uses two terms:

1. Accuracy on the [[Simulation Set]] after the rollout training steps.
2. Parameter count of the model after the action (smaller or larger models can be rewarded depending on weights).

Each term has its own grading curve. Exact formulas are still to be written here (TODO).

---

### Where it runs

Only inside simulation rollouts and expand steps, not in the main [[Training loop]]. Rollouts use one epoch per action with a constant learning rate, so the score reflects a short update, not full convergence.

---

### Comparison with the original growingNN paper

The paper uses simulation to rank architecture changes. The weighted multi-criteria score matches that idea. The split between accuracy and model size is an engineering choice in the original code.

---

### Known limitations

Scores depend on the small [[Simulation Set]] and on one-step training in rollouts. A move that looks good in simulation may still fail after a full generation of training.
