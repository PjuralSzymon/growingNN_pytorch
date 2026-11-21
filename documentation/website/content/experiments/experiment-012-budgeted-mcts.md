# Experiment 012: Budgeted search on CIFAR-10

## Result

The completed budgeted-search run grew the Experiment 010 starter from **296
to 2,432 parameters** and finished ten generations at **53.90% validation
accuracy and 54.14% official test accuracy**. It took about 73.8 minutes.

| Run | Completed seeds | Final validation | Final test | Final parameters |
| --- | --- | ---: | ---: | ---: |
| Budgeted search with widening, direct scoring (`v3`) | 1 | 53.90% | 54.14% | 2,432 |

This is one completed seed, not evidence that this search is better than the
original Monte Carlo algorithm or another approach. There is no completed
matched original-MCTS run in this experiment.

## What was tested

The run used the Experiment 010 starter: one 8-channel convolution, global
average pooling, and a linear classifier. It trained CIFAR-10 for ten generations
of twenty epochs each. Search used a 4,500-image simulation-training subset,
a 1,120-image simulation-validation subset, and a nominal 32-second budget.

This run used the experimental budgeted search with convolution widening
enabled. Its tree was limited to root actions and rollout depth was zero, so
candidate scores measured the next mutation after short simulation training.
It selected the final action by the candidate's mean score.

**The implementation did not guarantee that every root action was evaluated.**
It could stop after the deadline while actions remained. Evaluations already in
progress could overrun the budget. This differs from the requested rule that
all root actions must be evaluated.

## Training and model growth

![Experiment 012 training accuracy, validation accuracy, and parameter count](/assets/experiments/012-training-and-parameters.png)

The upper graph shows accuracy after each epoch; the lower graph shows the
number of parameters during training. Dotted lines mark generation boundaries.
The reported test accuracy is the final result, not an epoch-by-epoch curve.

## What happened

Validation accuracy is recorded at the end of each generation, before the
listed action is applied. The last generation has no following action.

| Generation | Validation | Parameters at end | Action selected for next generation |
| ---: | ---: | ---: | --- |
| 0 | 32.82% | 296 | Add a sequential convolution with BatchNorm and ReLU |
| 1 | 41.14% | 896 | Add a residual convolution |
| 2 | 46.66% | 1,480 | Add a sequential linear layer |
| 3 | 44.26% | 1,552 | Add dropout |
| 4 | 47.12% | 1,552 | Increase hidden linear width |
| 5 | 50.26% | 1,704 | Add dropout |
| 6 | 52.70% | 1,704 | Add a sequential linear layer |
| 7 | 51.96% | 1,776 | Add a sequential linear layer |
| 8 | 52.82% | 1,848 | Add a residual convolution |
| 9 | 53.90% | 2,432 | — |

Nine searches were recorded. They evaluated five to seven distinct candidates
each and took approximately 34–41 seconds. The records do not contain the total
number of legal root actions, so they cannot establish exact coverage. No
convolution-widening action was selected in the completed run.

## Earlier pilots and the widening defect

An earlier budgeted-only pilot was stopped during generation 3. A separate
widening pilot was stopped during generation 2 after a widening action dropped
immediate validation accuracy from **41.68% to 23.40%** on a graph with a
second convolution. Those partial runs are not comparable final results.

The widening implementation was revised to preserve outputs on the supported
unbranched convolution paths, and regression tests were added. A later structural
replay indicated widening was unavailable after a residual branch was added;
the separate Experiment 013 prototype explored support for that case.

## How the budgeted search worked

The budgeted search's code has since been removed from the repository; this
section records what it did. It was an alternative to the project's Monte Carlo
tree search with three changes:

- **One new candidate per iteration.** Each node kept a shuffled list of its
  untried actions. An iteration walked down the tree and, at the first node with
  untried actions, applied one of them, scored that single new child and
  stopped. Nodes with no untried actions were descended by UCB1 with the
  square-root formula (mean score plus `C * sqrt(ln N / n)`).
- **The deadline was a hard stop.** After the first evaluation, the search
  stopped as soon as the time budget expired, even if root actions were still
  untried. This kept every search close to its budget (34–41 seconds here), but
  is also why it could not guarantee that all root actions were scored.
- **Each score was added once along the path.** The new child's score was added
  to every node from the root to that child, so every node's value divided by
  its visits was a true mean. The final action was the root child with the
  highest mean, not the highest UCB1.

The run wrapped the search so it drew from its own random-number streams and did
not change the training run's random sequence, and it limited the tree to root
actions with rollout depth zero.

**What was kept.** Selecting by mean score, adding each score only once and
skipping random follow-up actions (rollout depth 0) are now part of Monte Carlo
2 (see [Experiment 015](experiment-015-action-replay.md)). The convolution
widening action used here is kept in the library. The hard-stop deadline was
not kept: Monte Carlo 2 still scores every root action at least once.

## Limits and follow-up

- Only seed 1 completed. There is no matched multi-seed comparison with original
  MCTS or other algorithms.
- Search reused part of the validation set. The official test set was reported
  separately and was not used to select actions.
- The board labels its simulation algorithm `none` because the experimental
  wrapper did not supply a name. The run path and history identify the method.
- The experiment's code is no longer in the repository. A source snapshot and
  fingerprint were saved with the `v3` run (`source_snapshot.zip` in the run's
  output folder, which is not tracked by git); use those to reproduce it.

[Experiment 013](experiment-013-family-search.md) reports the subsequent
family-balanced run separately. Its one-seed result is higher, but the runs
changed several things and neither enforces full root coverage.

## Saved results

- [Run summary](../../../../experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/run.json)
- [Experiment board](../../../../experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/board/main.json)
- [Training history](../../../../experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/history.pt)

The board's `generations`, `simulations`, and `graphs` directories contain
per-generation metrics, evaluated candidates, and architecture drawings.
