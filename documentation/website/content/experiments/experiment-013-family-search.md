# Experiment 013: Growing a small network on CIFAR-10

## Result

The family-balanced search grew the 296-parameter starter to 80,887 parameters
and reached **69.48% validation accuracy and 69.83% official test accuracy**
after ten generations. The earlier budgeted-search run reached **53.90%**
validation and **54.14%** test accuracy with 2,432 parameters.

| Run | Completed seeds | Final validation | Final test | Final parameters | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| Experiment 012: budgeted search | 1 | 53.90% | 54.14% | 2,432 | 73.8 min |
| Experiment 013: family-balanced search | 1 | 69.48% | 69.83% | 80,887 | 83.9 min |

The validation difference is **15.58 percentage points** for seed 1. This is
an encouraging result, but one seed is not enough to establish reliability.
The runs used different source versions and differ in more than one search
feature, so this comparison does not identify which change caused the gain.
The achieved accuracy is not yet the strong CIFAR-10 result sought by the goal.

## What was tested

Both runs started from the Experiment 010 model: one 8-channel convolution,
global average pooling, and a linear classifier, totaling 296 parameters. They
used CIFAR-10, seed 1, ten generations of twenty training epochs, and the same
training recipe. Each search requested 4,500 training images and 1,120
validation images for simulation, with a nominal 32-second search budget.

The family-balanced search differs by spreading early candidate evaluations
across action types and offering widening across supported residual branches.
Both compared runs used direct root-action scoring: no additional mutation was
applied during a rollout. Both could stop when the time budget expired **before
every root action was evaluated**. This is a limitation of these experimental
implementations and does not match the requested full-root-coverage rule.
An evaluation already in progress was allowed to finish after the deadline.

## Training and model growth

![Experiment 013 training accuracy, validation accuracy, and parameter count](/assets/experiments/013-training-and-parameters.png)

The upper graph shows accuracy after each epoch; the lower graph shows the
number of parameters during training. Dotted lines mark generation boundaries.
The reported test accuracy is the final result, not an epoch-by-epoch curve.

![Validation accuracy and parameter growth comparison for experiments 011 and 012](/assets/experiments/012-013-validation-and-growth.png)

The comparison uses a logarithmic parameter-count scale because the final
models differ greatly in size. It shows one seed for each implementation.

## What happened in the family-balanced run

Generation numbering here starts at **0**, so generation 3 ends at epoch 80.
The action is applied after the listed generation. "Immediate change" compares
validation just after and just before the action, without further training.
"After 20 epochs" compares the next generation's final validation with the
value before the action. Changes are in **percentage points** (pp).

| Generation ending | Validation before action | Parameters | Action | Immediate change | After 20 epochs |
| ---: | ---: | ---: | --- | ---: | ---: |
| 0 (epoch 20) | 32.82% | 296 | Widen convolution: 8 → 16 | 0.00 pp | **+0.98 pp** → 33.80% |
| 1 (epoch 40) | 33.80% | 592 | Widen convolution: 16 → 32 | 0.00 pp | **+3.10 pp** → 36.90% |
| 2 (epoch 60) | 36.90% | 1,184 | Widen convolution: 32 → 64 | 0.00 pp | **−3.16 pp** → 33.74% |
| 3 (epoch 80) | 33.74% | 2,368 | Add convolution + BatchNorm + ReLU | 0.00 pp | **+20.84 pp** → 54.58% |
| 4 (epoch 100) | 54.58% | 39,424 | Add a sequential linear layer | 0.00 pp | **+5.26 pp** → 59.84% |
| 5 (epoch 120) | 59.84% | 43,584 | Increase hidden linear width | −2.26 pp | **+1.94 pp** → 61.78% |
| 6 (epoch 140) | 61.78% | 44,034 | Decrease hidden linear width | −2.00 pp | **+2.04 pp** → 63.82% |
| 7 (epoch 160) | 63.82% | 43,509 | Increase hidden linear width | −0.26 pp | **+4.72 pp** → 68.54% |
| 8 (epoch 180) | 68.54% | 43,959 | Add a residual convolution | 0.00 pp | **+0.94 pp** → 69.48% |
| 9 (epoch 200) | 69.48% | 80,887 | No further action | — | — |

The 20-epoch changes describe what happened **after** each action; they are not
causal estimates of the action's benefit. The model also trained for 20 epochs,
the learning rate changed, and validation accuracy fluctuated. In particular,
the negative result after 32 → 64 widening does not prove the widening was
harmful, nor do later gains prove that each selected action helped. A trained
unchanged-model control would be needed to make those comparisons.

## The jump after epoch 80

At the end of generation 3, search selected `AddSeqConvLayer`. Before the
action, the network had a single 3→64 convolution followed by ReLU, global
average pooling, and the classifier. The action inserted a **64→64, 3×3
convolution**, followed by **BatchNorm and ReLU**, before global average
pooling. It increased the parameter count from **2,368 to 39,424**: the new
convolution has 36,928 parameters and BatchNorm has 128 trainable parameters.

The convolution was initialized as a spatial identity: its center weights on
the channel diagonal were one, and the other weights and biases were zero.
Immediately after insertion, full validation accuracy remained **33.74%**. The
large visible jump occurred **after training resumed**, not at the moment of
insertion:

| Point | Training accuracy | Validation accuracy | Parameters |
| --- | ---: | ---: | ---: |
| End of epoch 80, before action | 37.31% | 33.74% | 2,368 |
| Immediately after action, before training | — | 33.74% | 39,424 |
| End of epoch 81, after one training epoch | 42.34% | 46.50% | 39,424 |
| End of epoch 100, after 20 training epochs | 56.12% | 54.58% | 39,424 |

The extra convolution can learn spatial combinations of the 64 feature maps
before the model averages them into one value per channel. The preceding
widening steps also gave that convolution more channels to work with. These
are plausible reasons for the improvement, but the run does not establish how
much came from the new convolution, BatchNorm, ReLU, the earlier widening, or
ordinary continued training. The pre-action validation score was also a noisy
low point: epochs 78 and 79 recorded 38.28% and 36.08%.

In that search, this candidate received a **39.15% mean simulation-validation
score over two visits**; the next-highest recorded candidate scored 37.32% in
one visit. Each score followed one training epoch on the simulation training
subset and evaluation on the simulation validation subset, rather than 20
epochs on the full training set. The configured rollout depth was zero, so no
additional random action was added before scoring. The score is therefore a
short-term measurement, not a direct prediction of the 54.58% achieved after
the next full training generation. All five root candidates generated at this
particular decision were scored. The official test accuracy rose from **35.48%** at the end of
generation 3 to **55.31%** at the end of generation 4; this is supporting
evidence from the same single run, not an independent action-control experiment.

### How A08's score was calculated

Each visit trained a copy of the candidate for one simulation epoch, then
measured validation accuracy on the simulation subset (4,500 training and
1,120 validation examples). The configured rollout depth was zero: no further
random architecture action was added before this scoring training. Only
validation accuracy contributed to the score; loss, parameter count, and time
had weight zero. Thus the selection score was the average accuracy over visits.

| A08 evaluation | Simulation validation accuracy | Evidence |
| --- | ---: | --- |
| First | **39.1071%** | Reconstructed as `2 × mean − last` |
| Second | **39.1964%** | Last recorded accuracy / composite score |
| Mean used for selection | **39.1518%** | `(39.1071% + 39.1964%) / 2` |

The board retains the mean and the latest score, not a full evaluation history;
the first value above is an inference from those saved values. The visits were
separate evaluations of copies of the candidate, not consecutive epochs of
training on one model.

On the first visit A08 led the next candidate, widening to 256 channels
(37.3214%), by **1.7857 percentage points**. It led the convolution without
BatchNorm/ReLU (25.9821%) by **13.125 points**. This is a useful early signal,
especially against the other convolution, but one evaluation per alternative
does not establish which candidate would train best over 20 epochs. The two
convolution actions differ in more than the presence of BatchNorm alone, so
their scores cannot isolate its effect. The full-validation score before the
action is measured on a different set and cannot establish a simulation-subset
before/after gain.

The five generated candidates were each evaluated once. There was time for a
sixth evaluation. At that point all had one visit, so the UCB1 exploration
bonus was the same for each: `sqrt(ln(5) / 1) = 1.2686`, with exploration
coefficient 1. A08 had the highest first accuracy, so its UCB1 was about
`0.3911 + 1.2686 = 1.6597`, and it received the sixth evaluation. The final
choice was made using the highest **mean accuracy**, not UCB1.

The board's final A08 UCB1 value, **1.3380**, is a search priority calculated
after six visits: `0.3915 + sqrt(ln(6) / 2)`. It can exceed 1 because it is
not an accuracy. It was not added to A08's final selection score. Other
candidates had higher UCB1 values at that point because they had fewer visits,
but the time-limited search ended and selected the highest mean instead.

Nine searches were recorded. Each evaluated five distinct root candidates,
with five or six total evaluations, and took approximately 35–39 seconds. The
run log recorded 5 generated root actions in generations 0–3, 9 in generation
4, and 21 in each of generations 5–8; later searches therefore left candidates
unscored.

## Candidate scores by generation

The charts are stacked by generation with fixed action-type positions across
every row. A blank position means no candidate of that type was recorded as
evaluated; it does not mean a score of zero or prove that the action was
unavailable. Every distinct recorded action has a fixed slot and an `A` number
within its type; its location, dimensions or ratio are shown below its bar.
For widening, the per-evaluation random seed is excluded when matching actions
across generations. The [full recorded action details and scores](/assets/experiments/013-action-score-details.csv)
include the original seeds and parameters.

Each bar is one **evaluated root candidate**, grouped by action family. Its
height is the mean simulation-validation accuracy on a 0–1 scale; the star and
black border mark the selected action. `n` is its number of evaluations. Almost
all candidates were evaluated once; the board does not retain the individual
scores for candidates visited twice, so the bars do not show uncertainty or a
distribution of individual trials. For A08 specifically, its first of two
scores can be reconstructed from the saved mean and latest score, as shown
above.

The table counts the individual **scored candidates** shown as bars, not all
actions available to the search. The final columns show the total root actions
generated, as recorded in `growingnn.log`, and how many received no score.
The log does not record the full generated pool by type, so the zeros below do
not establish that an action type was unavailable.

| Generation | Widen conv | Seq conv with BN | Seq conv without BN | Residual conv | Seq linear | Add neurons | Delete neurons | Dropout | Scored | Generated | Unscored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 5 | 5 | 0 |
| 1 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 5 | 5 | 0 |
| 2 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 5 | 5 | 0 |
| 3 | 2 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 5 | 5 | 0 |
| 4 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 5 | 9 | 4 |
| 5 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 5 | 21 | 16 |
| 6 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 5 | 21 | 16 |
| 7 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 5 | 21 | 16 |
| 8 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 5 | 21 | 16 |

Both sequential-convolution groups use the same colour. In generations 0–3,
all five generated candidates were scored; in generations 5–8, only five of
21 were scored. Repeated evaluations of one candidate still produce one bar.

![Candidate scores for each search in Experiment 013](/assets/experiments/013-action-scores-overview.png)

Generation numbers start at zero. Generation 3 is the search **after epoch
80**. There is no generation-9 search because training ended after that
generation. Larger individual charts: [0](/assets/experiments/013-action-scores-gen-0.png),
[1](/assets/experiments/013-action-scores-gen-1.png),
[2](/assets/experiments/013-action-scores-gen-2.png),
[3](/assets/experiments/013-action-scores-gen-3.png),
[4](/assets/experiments/013-action-scores-gen-4.png),
[5](/assets/experiments/013-action-scores-gen-5.png),
[6](/assets/experiments/013-action-scores-gen-6.png),
[7](/assets/experiments/013-action-scores-gen-7.png), and
[8](/assets/experiments/013-action-scores-gen-8.png).

### Why the epoch-80 action won, and what happened later

![Candidate scores after epoch 80](/assets/experiments/013-action-scores-gen-3.png)

At generation 3, the convolution with BatchNorm and ReLU had the highest
recorded mean: **0.392 over two evaluations**, compared with **0.373** for the
next-highest candidate, widening to 256 channels (one evaluation). The
convolution without BatchNorm and ReLU scored **0.260** in one evaluation. The
search selected the candidate that scored highest among the five it evaluated.
Its short score does not directly quantify the 20.84-point later improvement;
that gain is measured after subsequent live training on a different evaluation
set, and there is no unchanged-model simulation score in the records.

| Later search | Recorded sequential-convolution candidate | Score | Selected candidate's score | Result |
| --- | --- | ---: | ---: | --- |
| Generation 4 | Add convolution with BatchNorm and ReLU after `seq_conv_0` | 0.496 | 0.562 | A linear insertion won. |
| Generation 5 | No sequential convolution in the evaluated records | — | 0.598 | Cannot tell whether one was available but not sampled. |
| Generation 6 | Add convolution with BatchNorm and ReLU between `conv1` and `seq_conv_0` | 0.626 | 0.657 | Neuron deletion won. |
| Generations 7–8 | Evaluated sequential convolutions had no BatchNorm/ReLU | 0.570 / 0.667 | 0.669 / 0.689 | Other actions won; no BatchNorm/ReLU candidate appears in these records. |

The later candidates are **not the identical edit**: the network already has
the first new convolution, and later candidates insert another layer at
different locations in a more-trained model. Absolute scores also rise as the
starting model improves, so compare candidates within a generation rather than
treating a higher later score as a larger benefit. The records do not show what
would have happened after 20 normal training epochs for an unchosen candidate.
The search did not guarantee full root coverage, and scores are mostly single
evaluations, so they cannot establish whether a better convolution was missed
or whether the short score reliably predicts long-term benefit. The saved
decision checkpoints can support a future controlled test.

## How the family-balanced search worked

The family-balanced search's code has since been removed from the repository;
this section records what it did and which ideas were kept. It scored root
actions only (rollout depth 0), like Experiment 012, with three additions:

- **Family round-robin order.** Candidates were grouped by action type
  (widening, sequential convolution, residual convolution, linear layer,
  dropout, neuron changes, and so on). The order was randomized within each
  type and across types, then candidates were taken one per type in turn. When
  the budget could not cover every action, this spread the scored candidates
  across action types instead of spending the budget on many variants of one
  type. After every candidate had a score, the remaining time went to UCB1
  re-evaluations, and the final action was the highest mean score.
- **Fan-out widening.** Besides normal widening, it offered widening of a
  convolution whose output feeds several convolutions (for example after a
  residual branch), through ReLU, dropout or identity operations. Each
  downstream convolution's input weights were split between the duplicated
  channels so the network's output did not change.
- **Decision checkpoints.** Before each search it saved the model, input shape
  and learning-rate schedule, so a decision could be replayed later. Experiment
  14 replays the generation-3 checkpoint saved by this run.

**What was kept.** Both widening variants are kept in the library as one
action, `WidenConv`: it widens along a single path or, when the output feeds
several convolutions, along every branch. Experiment 15 Phase 7 ran before fan-out
widening was added to the library and found that no widening was available once a
residual branch was added after `conv1`. Monte Carlo 2 keeps selecting by mean
score with rollout depth 0. Family round-robin ordering only matters when the
budget cannot cover every root action; Monte Carlo 2 scores every root action at
least once, so it was not kept.

## Status and limitations

- Only seed 1 completed. A seed-2 run was interrupted before its first search;
  seed 3 was not started. Its board may still display `running` because the
  process was interrupted.
- The budgeted and family-balanced implementations did not guarantee scoring
  all root actions. Do not treat this run as a test of full-root coverage.
- Simulation uses part of the validation set repeatedly. The official test
  result is reported separately and was not the selection metric for actions.
- The family-balanced run changed candidate ordering and the available action
  set together. An ablation and more seeds are needed to assess each change.

## Saved results

- [Family-balanced run summary](../../../../experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1/run.json)
- [Family-balanced board](../../../../experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1/board/main.json)
- [Budgeted baseline summary](../../../../experiments/output/train_cifar10/runs/exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1/run.json)
- Source code: `source_snapshot.zip` in `experiments/output/train_cifar10/runs/exp012_family_search/v1/` (not tracked by git)

The board's `generations`, `simulations`, and `graphs` directories contain
per-generation metrics, evaluated candidates, and architecture drawings.
