The scoring function judges how good a model is after a simulated action sequence. 
### Score terms

The user defines a weighted sum of sub-scores. The original setup uses two terms:

1. Accuracy on the [[Simulation Set]] after the rollout training steps.
2. Parameter count of the model after the action (smaller or larger models can be rewarded depending on weights).

Each term has its own grading curve. Exact formulas are still to be written here (TODO).
