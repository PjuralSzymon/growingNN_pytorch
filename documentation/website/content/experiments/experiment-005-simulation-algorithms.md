# Experiment 005: Simulation algorithm comparison

We keep the best learning-rate package from Experiment 004 (`composed_exponential`). We change only the simulation search algorithm and the starter architecture.

MCTS (`montecarlo`) can look ahead with rollouts, but it is unstable across seeds. The goal of this experiment is to find a simulation algorithm that stays stable and still looks beyond the next action, instead of locking onto one lucky local step.

Script: `experiments/train_mnist_exp005_simulation_algorithms.py`

Charts: `documentation/website/scripts/generate_experiment_005_charts.py`

Folder: `experiments/output/train_mnist/runs/exp005_simulation_algorithms`

Snapshot: `documentation/website/data/experiments/experiment-005-simulation-algorithms.json`

This page is a live report. Tables and charts use only boards with `status=completed` (`130` / `130` cells, `100.0%`).

Color scheme on charts: blue = `big` starter, green = `medium` starter. Pooled scores that mix both starters use purple, not green or blue.

## Experiment parameters

| Parameter | Values | Purpose |
| --- | --- | --- |
| Simulation algorithm | thirteen IDs below | Find a stable search method that can look ahead |
| Starter | `big`, `medium_1conv_2linear` | Easy vs harder growth ladder |
| Seed | `100`, `101`, `102`, `103`, `104` | Five matched seeds per algorithm × starter |

| Fixed parameter | Value | Explanation |
| --- | ---: | --- |
| Dataset | MNIST | Classification task |
| Planned cells | `130` | `13` algorithms × `2` starters × `5` seeds |
| Completed cells in this refresh | `130` | Partial grid is still valid for live reading |
| LR package | `composed_exponential` × logistic recovery | Best package from Experiment 004 |
| Effective LR rule | `max(0.001, base_lr(epoch) * recovery_factor)` | Global exponential base times action recovery |
| Standard cell `lr_alpha` | `0.01` | Target / peak learning rate |
| Minimum LR floor | `0.001` | Hard floor on optimizer LR |
| Exponential gamma | `0.98` | Base decay for `composed_exponential` |
| Recovery warmup | logistic | Shape after an architecture action |
| Warmup length | `10` | Scheduler iterations after an action |
| Warmup steepness `k` | `10` | Logistic shape parameter |
| Simulation score object | `SimulationScore` | Weighted sum of score terms |
| Accuracy score term | `score_by_learning.score_acc` | Default learning grade |
| Accuracy metric | `val_acc` | Uses validation accuracy after short GD |
| How `score_acc` trains | `run_simulation_scoring_gradient_descent` | Short GD on sim loaders, then read last metric |
| Simulation training epochs | `15` | Epochs inside that scoring GD |
| Simulation set size | `2000` | Samples used by simulation scoring |
| Score weight accuracy | `1.0` | `score_weight_acc` |
| Score weight parameter count | `0.1` | `score_weight_countw` |
| Slope threshold | `3°` | `SlopeEstimationSimulationScheduler` gate |
| Simulation scheduler | `SlopeEstimationSimulationScheduler` | Runs search when the training slope is flat enough |
| Generations | `10` | Architecture-decision cycles |
| Epochs per generation | `10` | Recorded epochs per generation |
| Total training epochs | `100` | `10 × 10` |
| Batch size | `64` | Training samples per batch |
| Simulation time | `120 s` | Wall-time budget inside one search call |
| Model channels | `4` | Conv channel width for both starters |
| Hidden linear size | `16` | Linear width for both starters |
| Big starter | `BigAvgPoolMnistNet` | `2×Conv + 2×Linear`, start params `420` |
| Medium starter | `Medium1Conv2LinearMnistNet` | `1×Conv + 2×Linear`, start params `276` |
| Deterministic seeding | on | `configure_deterministic_seeding()` |

Shared scoring note: almost every algorithm grades a candidate by calling `running_config.simulation_score.score(...)`. When the accuracy weight is on, that call runs short gradient descent through `score_by_learning.score_acc` → `run_simulation_scoring_gradient_descent`. So “score” already means “train briefly on the simulation set, then read accuracy.” MCTS adds extra short GD inside expand/rollout before that shared score. See the `montecarlo` section.

Run path:

```text
exp005_simulation_algorithms/<simulation_alg_id>/<model_name>/<hp_folder>/seed_<seed>/
```

## Research questions

Main question: which simulation algorithm is stable across seeds and still searches beyond one local step?

Supporting checks:

1. Which algorithm gives high final accuracy with low seed variance?
2. Is the simulation algorithm good for both the big and the medium starter?
3. Is the impact of simulation actions stable across generations, or does almost all useful gain come from the first live action?

## Result timeline

Progress in this refresh: `130` / `130` completed (`100.0%`).

| Algorithm | Big | Medium | Notes |
| --- | ---: | ---: | --- |
| `montecarlo` | `5` / `5` | `5` / `5` | Finished both |
| `greedy` | `5` / `5` | `5` / `5` | Finished both |
| `random` | `5` / `5` | `5` / `5` | Finished both |
| `sequential_halving` | `5` / `5` | `5` / `5` | Finished both |
| `ugape` | `5` / `5` | `5` / `5` | Finished both |
| `successive_rejects` | `5` / `5` | `5` / `5` | Finished both |
| `beam_search` | `5` / `5` | `5` / `5` | Finished both |
| `best_first` | `5` / `5` | `5` / `5` | Finished both |
| `shot` | `5` / `5` | `5` / `5` | Finished both |
| `sequential_halving_beam` | `5` / `5` | `5` / `5` | Finished both |
| `ugape_deepen` | `5` / `5` | `5` / `5` | Finished both |
| `progressive_widening` | `5` / `5` | `5` / `5` | Finished both |
| `hierarchical_search` | `5` / `5` | `5` / `5` | Finished both |

## Algorithms used

Descriptions follow the code in `growingnn/simulation/simulation_algorithms/`.

Shared words used below:

- Current live model: the model that entered this simulation call. Search starts from that model.
- Root action: a legal architecture mutation from the current live model. Depth of the action tree starts at `1` here.
- Depth-1 search: the algorithm only compares those next mutations. It does not apply a second future mutation before choosing.
- Look-ahead: the algorithm can apply further mutations after the first one before it chooses which root action to return. In this experiment the look-ahead hybrids use max depth `2` (one root action, then one more future action).
- Arm: one candidate in a race. For depth-1 methods, one arm = one root action plus the child model after that action was applied once.
- Living arm: an arm that has not been eliminated yet.
- Rescore: call `score_fn` again on the same child model. Because scoring runs short GD, each rescore can return a slightly different noisy reading. The algorithm updates a running mean. This is not look-ahead and not a new architecture step. Seed fact: each rescore starts from the same child weights (`deepcopy` of that child), but the search does not reset the experiment RNG seed before every rescore, so SGD noise can differ between rescored readings.
- Beam: the short list of best current candidates kept for further expansion. Beam width is how many candidates stay on that list.

### `random`

Picks one legal root action at random. No scoring.

```text
actions = generate_all_actions(model)
return random.choice(actions)
```

Search depth: depth-1 only. No look-ahead.

### `greedy`

Tries root actions one by one until the wall-time budget ends. Each try: copy the model, apply the action, score it with shared simulation scoring GD, keep the highest score.

```text
remaining = all root actions
while time left and remaining:
  action = random pick from remaining
  score = score(apply(copy(model), action))
  keep best score / action
return best_action
```

Search depth: depth-1 only. No look-ahead.

### `sequential_halving`

This algorithm only searches depth `1` of the action tree. It does not look ahead.

Definitions for this method:

- Arm: one root action and the child model created by applying that action once.
- Living arms: the arms still in the race.
- Mean: running average of repeated scores of the same child.
- Why rescoring: shared `score_fn` trains briefly with GD, so one score is noisy. Repeating the score on the same child averages that noise. It does not apply a second architecture action. Same-child start weights each time; experiment seed is not reset on every rescore, so readings can still differ.

Steps:

1. Build one arm per root action. Expand once: copy, apply the action, keep the child. Do not score yet in the expand loop; scoring starts in the race.
2. While more than one living arm remains and time remains: score every living arm once more, update each arm’s mean, sort by mean, keep only the top half (`ceil(n / 2)`).
3. Return the last surviving arm with the best mean.

```text
for each root action: create arm = apply(action) once
living = all arms
while |living| > 1 and time left:
  for each living arm: score same child again; update mean
  keep top ceil(|living| / 2)
return living arm with best mean
```

Important: after the first half is kept, the next round scores those same children again. It is not deepening the tree. It is not look-ahead.

### `ugape`

Also depth-1 only. Same arm idea as Sequential Halving: one arm per root action, child created once, then repeated rescores of the same child.

After every root arm has been rescored once (when time allows), the loop repeatedly chooses between:

- the current best mean arm, and
- the strongest challenger

Strongest challenger means: among all other scored arms, the arm with the highest upper confidence bound

`mean + UGAPE_C * sqrt(max(log(total_score_calls), 1) / n)` with `UGAPE_C = 1.0`.

Then it rescores the contested arm that has fewer samples so far.

```text
for each root action: create arm = apply(action) once
rescore each arm once
while time left:
  best = max mean
  challenger = max upper_bound among the others
  rescore the one with fewer samples
return arm with best mean
```

Search depth: depth-1 only. No look-ahead.

### `successive_rejects`

Depth-1 elimination with a fixed rescore schedule from the code (`_log_bar`, `nk`).

Same arm idea: one arm per root action, child applied once, then repeated scoring of that child.

1. Create one arm per root action.
2. For rounds `1 .. n-1`: give each living arm a scheduled number of extra rescored readings, then drop the current worst mean (reject one arm).
3. Return the best remaining mean.

```text
create all arms
for round in 1 .. n-1 while time and |living| > 1:
  rescored_each = schedule(round)
  score each living arm rescored_each times
  drop worst by mean
return best mean among living
```

Search depth: depth-1 only. No look-ahead.

### `montecarlo`

Monte Carlo Tree Search over architecture mutations. This one does look ahead.

Shared scoring still uses short GD through `SimulationScore`. On top of that, MCTS runs its own short `gradient_descent` inside `TreeNode.expand` and `TreeNode.rollout` with `MCTS_ROLLOUT_EPOCHS` and `MCTS_ROLLOUT_LR`. Then the leaf is graded with `simulation_score.score`, which can train again. So MCTS can train more than depth-1 methods: rollout GD plus scoring GD.

Steps:

1. Start from a root `TreeNode` holding the current model.
2. Until time is up: select children with UCB1, expand missing actions (each new child gets the MCTS short GD), evaluate leaves with `TreeNode.rollout()`.
3. A rollout applies up to `MCTS_ROLLOUT_DEPTH = 2` random future actions. After each future action it runs short MCTS GD. Then it calls shared `simulation_score.score`.
4. Return the root child with the best UCB1 value.

Constants: `MCTS_UCB1_C = 2`, `MCTS_ROLLOUT_DEPTH = 2`, `MCTS_ROLLOUT_EPOCHS = 1`, rollout LR `0.0001`.

```text
root = TreeNode(model)
while time left:
  simulate(root)   # UCB1 select, expand, rollout future actions, backprop
return root.get_best_child().action
```

Search depth: look-ahead through rollouts.

### `beam_search`

Look-ahead search with a fixed shortlist size.

Definitions:

- Beam width (`BEAM_WIDTH = 3`): keep only the top 3 scored candidates.
- Deepen: from each beam candidate, generate further legal actions, apply one more mutation, score the new child.
- Max depth (`MAX_DEPTH = 2`): after the root action (depth 1), allow one more action (depth 2).
- Returned choice: the root action that led to the best scored node found anywhere in the search.

Steps:

1. Score every root action (copy, apply, shared score).
2. Keep the top 3 as the beam.
3. While time remains and current beam depth is below 2: expand every beam node with all further actions, score children, keep the new top 3.
4. Track the globally best scored node. Return that node’s root action.

```text
frontier = score all root actions
beam = top 3
while time and depth < 2:
  expand all actions from each beam node; score
  beam = top 3 children
return best.root_action
```

Search depth: look-ahead to depth 2.

### `best_first`

Priority-queue search. It prefers higher score and slightly prefers shallower depth (`DEPTH_PENALTY = 1e-3`).

`MAX_EXPANSIONS = 32` is a hard cap on how many nodes get expanded after the root scoring pass. It stops early so the 120 s budget is not spent expanding every possible child of every promising node. Depth is still capped at `MAX_DEPTH = 2`.

Steps:

1. Score all root actions and push them onto an open heap.
2. While the heap is not empty, time remains, and expansions `< 32`: pop the best node; if depth is already 2, skip; else expand children, score them, push them, update the global best.
3. Return the root action of the best scored node found.

```text
push all scored root nodes onto open heap
while heap and time and expansions < 32:
  pop best; if depth >= 2: skip
  expand children; score; push; update global best
return best.root_action
```

Search depth: look-ahead to depth 2, with expansion cap 32.

### `shot`

SHOT means Sequential Halving applied recursively on a short action tree (`MAX_DEPTH = 2`).

Definitions:

- Short tree: from the live model, allow at most two mutation steps.
- At a node with depth left: create one arm per legal action from that node.
- “Running SHOT one level deeper”: to score an arm, recursively run the same halving procedure on that child with `depth_left - 1`. When depth left is 0, just call shared `score_fn`.
- The set / living arms: the arms still kept after each halving round at that node.

```text
def shot(node, depth_left):
  if depth_left <= 0: return score(node)
  arms = expand all actions
  while |living| > 1:
    for arm: value = shot(arm.child, depth_left - 1)
    keep top half
  return best living value
run at root with depth_left = 1
```

Search depth: look-ahead through recursive halving.

### `sequential_halving_beam`

Hybrid of depth-1 Sequential Halving and beam look-ahead.

1. Spend about half the time (`ROOT_TIME_FRACTION = 0.5`) doing Sequential Halving on root arms (same rescoring idea as depth-1 Sequential Halving).
2. Take top survivors (`BEAM_WIDTH = 3`) and beam-deepen them to `MAX_DEPTH = 2`.
3. Return the root action of the best scored deepened node.

```text
SH on root until 0.5 * sim_time
beam = top 3 survivors
beam-deepen until deadline / depth 2
return best.root_action
```

Search depth: look-ahead in the second phase.

### `ugape_deepen`

Hybrid of depth-1 UGapE and beam look-ahead.

1. Run UGapE rescoring on root arms for half the time.
2. Take top `RIVAL_COUNT = 2` arms by mean.
3. Beam-deepen them (`BEAM_WIDTH = 3`, `MAX_DEPTH = 2`).
4. Return the root action of the best scored deepened node.

```text
UGapE rescores on root for 0.5 * sim_time
focus = top 2 by mean
beam-deepen focus
return best.root_action
```

Search depth: look-ahead in the second phase.

### `progressive_widening`

Does not open every root action at once.

1. Start with `INITIAL_OPEN = 3` unlocked root actions.
2. Unlock one more from a fixed closed list every `UNLOCK_EVERY_SEC = 5.0` seconds.
3. On the open set, run Sequential Halving-style pruning (rescoring means).
4. Limited beam deepen on survivors (`BEAM_WIDTH = 3`, `MAX_DEPTH = 2`).
5. Return the best root action found.

```text
unlock first 3 actions
while time and open set:
  maybe unlock one every 5 s
  SH-style prune open set
  beam-expand survivors one depth
return best_action
```

Search depth: look-ahead in the deepen phase.

### `hierarchical_search`

Two-stage family filter, then Sequential Halving + beam.

Definitions:

- Family: action class name (`type(action).__name__`), for example all residual-convolution actions in one family.
- Family time fraction (`FAMILY_TIME_FRACTION = 0.25`): first quarter of the budget grades families.
- Top families (`TOP_FAMILIES = 2`): keep only the two best families after that sample.

Steps:

1. Group root actions by class name.
2. Score one sample action per family during the family budget.
3. Keep the top 2 families.
4. Pool all actions from those families, run Sequential Halving, then beam deepen (`BEAM_WIDTH = 3`, `MAX_DEPTH = 2`).
5. Return the best root action.

```text
group actions by class name
score one sample per family
keep top 2 families
SH + beam-deepen on that pool
return best.root_action
```

Search depth: look-ahead in the deepen phase. No completed seeds yet in this refresh.

### Families by behavior

1. Baselines: `random`, `greedy`
2. Depth-1 only (next action only): `sequential_halving`, `ugape`, `successive_rejects`
3. MCTS with rollouts: `montecarlo`
4. Explicit multi-step search: `beam_search`, `best_first`, `shot`
5. Root ranking + deepen hybrids: `sequential_halving_beam`, `ugape_deepen`, `progressive_widening`, `hierarchical_search`

## Final accuracy

This section pools completed big and medium runs. The chart is sorted by mean final training accuracy across big and medium together. Lower mean train is on the left. Higher is on the right. The table is sorted the other way: best mean train on top.

![Final accuracy combined](/assets/experiments/005-final-accuracy-combined.png)

> [!CAPTION] Figure 1. Mean final train and validation accuracy (%) on big (blue) and medium (green). Sorted by mean train across completed big+medium runs.

| Algorithm | Seeds | Mean train (%) | Mean val (%) | Big train (%) | Big val (%) | Medium train (%) | Medium val (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sequential_halving_beam` | `10` | `85.80` | `87.46` | `87.24` | `88.60` | `84.37` | `86.31` |
| `ugape_deepen` | `10` | `85.66` | `87.45` | `87.11` | `88.55` | `84.20` | `86.34` |
| `best_first` | `10` | `84.78` | `86.95` | `85.54` | `87.71` | `84.03` | `86.20` |
| `progressive_widening` | `10` | `84.55` | `86.52` | `87.30` | `89.34` | `81.80` | `83.70` |
| `beam_search` | `10` | `84.36` | `86.76` | `87.87` | `89.90` | `80.86` | `83.62` |
| `ugape` | `10` | `83.97` | `86.74` | `85.27` | `87.89` | `82.67` | `85.59` |
| `greedy` | `10` | `83.52` | `87.09` | `87.99` | `88.91` | `79.05` | `85.27` |
| `sequential_halving` | `10` | `83.17` | `85.35` | `86.52` | `88.79` | `79.83` | `81.91` |
| `hierarchical_search` | `10` | `81.52` | `85.09` | `85.21` | `87.26` | `77.83` | `82.92` |
| `montecarlo` | `10` | `70.59` | `76.53` | `69.80` | `76.53` | `71.37` | `76.53` |
| `successive_rejects` | `10` | `66.42` | `67.54` | `45.37` | `45.53` | `87.48` | `89.55` |
| `shot` | `10` | `60.89` | `62.13` | `45.37` | `46.14` | `76.42` | `78.12` |
| `random` | `10` | `44.41` | `49.28` | `44.65` | `49.71` | `44.16` | `48.86` |

Best three by mean train: `sequential_halving_beam`, `ugape_deepen`, `best_first`.

Worst three by mean train: `random`, `shot`, `successive_rejects`.

Greedy is high. That is a warning: the current MNIST growth task is still easy. Use this grid as a filter. Keep methods that beat greedy and random, then re-test them on harder models later.

### Final accuracy on big

![Final accuracy on big](/assets/experiments/005-final-accuracy-by-algorithm.png)

> [!CAPTION] Figure 2. Mean final train and validation accuracy (%) on `big`. Bars = means. Gray circles = train seeds. Gray diamonds = validation seeds.

MCTS mean on big looks low even though many seed points sit high. The mean is correct: several big seeds land near `86.47`–`88.88`% train when high, but collapse seeds pull the mean down (mean `69.80`%, max `88.88`%). Gray seed markers make those low points easier to see against the mean bars.

### Final accuracy on medium

![Final accuracy on medium](/assets/experiments/005-final-accuracy-by-algorithm-medium.png)

> [!CAPTION] Figure 3. Mean final train and validation accuracy (%) on `medium`. Bars = means. Gray circles = train seeds. Gray diamonds = validation seeds.

On medium alone, `successive_rejects` has the best mean train. On the pooled table above it can still rank much lower if its big runs collapse.

## Seed stability

Mean alone is not enough. A good method should keep matched seeds close.

Charts sort unstable methods on the left and tighter methods on the right. Blue = big. Green = medium.

![Seed scatter all](/assets/experiments/005-seed-stability-final-val.png)

> [!CAPTION] Figure 4. Final validation seeds for all completed runs. Blue = big. Green = medium. Orange = mean.

![Composite all](/assets/experiments/005-composite-score.png)

> [!CAPTION] Figure 5. Composite score for all completed runs: mean final validation minus `0.15 * sqrt(variance)`. Left = worse. Right = better. Purple = pooled score across starters.

Best three by composite: `sequential_halving_beam`, `ugape_deepen`, `best_first`.

Worst three by composite: `random`, `shot`, `successive_rejects`.

Cross-check with final accuracy:

- The same top band leads both sections. Accuracy and stability agree so far.
- `sequential_halving_beam` leads composite in this refresh.
- `successive_rejects` stays weak on pooled composite even when medium accuracy looks strong.
- `shot` and `random` stay rejected on both accuracy and stability.

## Does the first action do all the work?

For each live architecture action we measure validation accuracy at the end of that generation versus the end of the next generation. Then we ask: of all positive recovered gains, how much comes from action order `1`?

We prefer methods that do not rely on one lucky first action. Smaller first-action share is better. Those methods are on the right.

Orange bars mark the current top three by mean final train, so we can see where the accuracy leaders sit on this axis.

![First-action share](/assets/experiments/005-first-action-gain-share-by-algorithm.png)

> [!CAPTION] Figure 6. Share of positive recovered validation gain from the first live action. Left = first action does almost everything. Right = later actions still matter. Orange = current top-3 by mean final train.

Best three here (smallest first-action monopoly): `montecarlo`, `random`, `successive_rejects`.

Worst three here (largest first-action monopoly): `progressive_widening`, `ugape_deepen`, `greedy`.

Cross-check with final accuracy:

- The methods with the smallest first-action share are not the accuracy leaders. A low first-action share alone does not make a winner.
- Accuracy leaders can still be among the worst here: most of their useful gain may come from the first live action.
- `sequential_halving_beam` sits nearer the middle than the worst first-action group.
- Among methods that stay high on accuracy and are less first-action-only than greedy, `beam_search` and `best_first` are the more interesting middle ground.

## Action composition

This section asks whether the algorithm is doing real exploratory work. We want a search method that uses more of the action set, not only one mutation type every time.

Charts sort low diversity on the left and higher diversity on the right.

![Action composition all](/assets/experiments/005-action-composition-by-algorithm.png)

> [!CAPTION] Figure 7. Executed action-type counts across all completed seeds. Left = mostly one type. Right = more mixed.

Best three by action mix (more types): `montecarlo`, `random`, `ugape`.

Worst three by action mix (almost one type): `successive_rejects`, `shot`, `progressive_widening`.

Cross-check with final accuracy:

- A top accuracy method can still be one of the worst on exploration. That weakens it as a final choice until harder tasks are tested.
- `successive_rejects` stays rejected as a general default when narrow mix pairs with bad pooled stability.
- `best_first` and `beam_search` sit in a useful middle: competitive final accuracy and more than one action family.

## Recovery after architecture actions

When the network changes, training accuracy can drop for a short time. This section asks whether training recovers after the mutation.

How we measure it:

1. Immediate change: training accuracy at the next epoch after the action, minus training accuracy at the end of the generation where the action ran.
2. Recovered change: training accuracy after one full recovery generation (`10` epochs of normal training), minus the same pre-action end-of-generation point.

Figure 8 shows mean values of those two changes per algorithm. Orange is the immediate shock. Blue is the recovered change after one generation. Stronger recovery sits on the right.

![Immediate vs recovered](/assets/experiments/005-action-impact-immediate-vs-recovered.png)

> [!CAPTION] Figure 8. Mean training-accuracy change after an architecture action. Orange = next epoch. Blue = after one recovery generation (`10` epochs).

Best three by recovered train change: `sequential_halving_beam`, `ugape`, `best_first`.

Worst three by recovered train change: `random`, `shot`, `successive_rejects`.

Cross-check with final accuracy:

- Recovery leaders are `sequential_halving_beam`, `ugape`, and `best_first`. Final-accuracy leaders are `sequential_halving_beam`, `ugape_deepen`, and `best_first`.
- The overlap is partial. Recovery still rejects the weakest absolute methods, but it is not a perfect copy of the accuracy ranking.
- Worst recovery: `random`, `shot`, `successive_rejects`.

## Starter effect

The starter can flip the ranking. Here we compare mean final training accuracy on big versus medium.

Sorted by absolute gap `|big − medium|`. Left = larger gap, more sensitive to model size. Right = smaller gap, more consistent across the two starters.

![Train big vs medium](/assets/experiments/005-final-train-big-vs-medium-by-algorithm.png)

> [!CAPTION] Figure 9. Mean final training accuracy on big (blue) versus medium (green). Sorted by absolute starter gap. Left = larger gap. Right = more consistent across model sizes.

Best three for starter consistency (smallest gap): `random`, `best_first`, `montecarlo`.

Worst three for starter consistency (largest gap): `successive_rejects`, `shot`, `greedy`.

Cross-check with final accuracy:

- `successive_rejects` again looks good on medium and bad on big. A large starter gap means the method is tied to model size.
- `best_first` is the useful consistency result among strong methods: high accuracy and a small big/medium gap.
- `sequential_halving_beam` and `ugape_deepen` also stay close across starters.

## Training histories

All simulation algorithms are shown in one grid. Blue = big. Green = medium.

![Training curves all](/assets/experiments/005-training-curves-all-algorithms.png)

> [!CAPTION] Figure 10. Training accuracy histories for every algorithm. Blue = big. Green = medium.

![Training mean fit](/assets/experiments/005-training-curves-mean-std-best-fit.png)

> [!CAPTION] Figure 11. Training mean ± std for every algorithm. Solid = mean. Band = ±1 std. Blue = big. Green = medium.

Best three looking training shapes: `sequential_halving_beam`, `ugape_deepen`, `best_first`.

Worst three looking training shapes: `random`, `shot`, `successive_rejects`.

Cross-check with final accuracy: the history shapes match the final ranking. Greedy stays high. That is still an easy-task warning, not a reason to pick greedy as the default.

## Validation histories

![Validation curves all](/assets/experiments/005-validation-curves-all-algorithms.png)

> [!CAPTION] Figure 12. Validation accuracy histories for every algorithm. Blue = big. Green = medium.

![Validation mean fit](/assets/experiments/005-validation-curves-mean-std-best-fit.png)

> [!CAPTION] Figure 13. Validation mean ± std for every algorithm. Solid = mean. Band = ±1 std. Blue = big. Green = medium.

Best three looking validation shapes: `sequential_halving_beam`, `ugape_deepen`, `greedy`.

Worst three looking validation shapes: `random`, `shot`, `successive_rejects`.

Cross-check with final accuracy: same high band and same rejects.

## Curiosity check: look-ahead vs depth-1

This is an extra group comparison, not a winner rule by itself. Depth-1 group = `random`, `greedy`, `sequential_halving`, `ugape`, `successive_rejects`. Look-ahead / hybrid group = the rest with completed seeds.

![Look-ahead vs depth-1](/assets/experiments/005-lookahead-vs-depth1-final-accuracy.png)

> [!CAPTION] Figure 14. Mean final train and validation accuracy for depth-1 methods versus look-ahead / hybrid methods. Bars use pooled completed runs in each group.

In this refresh the look-ahead / hybrid group has the higher pooled mean. That supports the main goal, but it is not enough alone: some look-ahead methods (`shot`, unstable `montecarlo`) are still weak, and some depth-1 methods (`ugape`, greedy) stay competitive on this easy task.

## Summary of results

We start from the final accuracy ranking, then walk section by section. After each point we update the shortlist of simulation algorithms that still look worth keeping.

Green-plus matrix below: one row per algorithm, one column per analysis section, in section order. A green plus means the algorithm was in the top 4 for that section. Rows are sorted by total green pluses (highest first). Tie-break uses final-accuracy rank.

| Algorithm | Final accuracy | Seed stability | First-action | Action mix | Recovery | Starter effect | Train histories | Val histories | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `best_first` | :g+: | :g+: |  |  | :g+: | :g+: | :g+: | :g+: | 6 |
| `sequential_halving_beam` | :g+: | :g+: |  |  | :g+: |  | :g+: | :g+: | 5 |
| `ugape_deepen` | :g+: | :g+: |  |  |  |  | :g+: | :g+: | 4 |
| `ugape` |  |  |  | :g+: | :g+: | :g+: |  |  | 3 |
| `greedy` |  | :g+: |  | :g+: |  |  |  | :g+: | 3 |
| `montecarlo` |  |  | :g+: | :g+: |  | :g+: |  |  | 3 |
| `random` |  |  | :g+: | :g+: |  | :g+: |  |  | 3 |
| `progressive_widening` | :g+: |  |  |  |  |  | :g+: |  | 2 |
| `beam_search` |  |  |  |  | :g+: |  |  |  | 1 |
| `successive_rejects` |  |  | :g+: |  |  |  |  |  | 1 |
| `shot` |  |  | :g+: |  |  |  |  |  | 1 |
| `sequential_halving` |  |  |  |  |  |  |  |  | 0 |
| `hierarchical_search` |  |  |  |  |  |  |  |  | 0 |

**1. Final accuracy**

Best three: `sequential_halving_beam`, `ugape_deepen`, `best_first`.
Close behind: `best_first`, `beam_search`.
Worst three: `random`, `shot`, `successive_rejects`.
Greedy is high, so the MNIST task is still easy.
Complete on both starters in this refresh: `sequential_halving_beam`, `ugape_deepen`, `progressive_widening`, `hierarchical_search`.
`progressive_widening` is complete on both starters.
Shortlist after this section: `sequential_halving_beam`, `ugape_deepen`, `best_first`, plus watch `best_first` and `beam_search`.

**2. Seed stability**

Best three by composite: `sequential_halving_beam`, `ugape_deepen`, `best_first`.
Worst three by composite: `random`, `shot`, `successive_rejects`.
The accuracy leaders also stay near the top on composite, so high means are not just lucky seeds.
Shortlist after this section: keep `sequential_halving_beam`, `ugape_deepen`, `best_first`. Keep watching `best_first` and `beam_search`. Reject `random`, `shot`, and `successive_rejects`.

**3. First-action effect**

Best three here (least first-action heavy): `montecarlo`, `random`, `successive_rejects`.
Worst three here (most first-action heavy): `progressive_widening`, `ugape_deepen`, `greedy`.
So some accuracy leaders look weak on this check: most useful gain can still come from the first live action.
`sequential_halving_beam` sits nearer the middle than the worst first-action group.
`best_first` and `beam_search` stay useful because they keep high accuracy without sitting in the worst first-action group.
Shortlist after this section: keep `sequential_halving_beam`, `best_first`, and `beam_search`. Keep `progressive_widening` and `ugape_deepen` with a first-action warning.

**4. Action composition**

Best three by mix: `montecarlo`, `random`, `ugape`.
Worst three by mix: `successive_rejects`, `shot`, `progressive_widening`.
Narrow mix weakens a method as a locked default even if accuracy is high.
`best_first` and `beam_search` sit in the useful middle: good accuracy and more than one action family.
Shortlist after this section: `sequential_halving_beam`, `best_first`, `beam_search` look best when accuracy and exploration are weighed together.

**5. Recovery after architecture actions**

Best three by recovered train change: `sequential_halving_beam`, `ugape`, `best_first`.
Worst three: `random`, `shot`, `successive_rejects`.
Recovery does not promote a new global winner.
Shortlist after this section: unchanged in spirit.

**6. Starter effect**

Best three for consistency: `random`, `best_first`, `montecarlo`.
Worst three for consistency: `successive_rejects`, `shot`, `greedy`.
Among strong methods, `best_first` remains a clear both-starter survivor. `sequential_halving_beam` and `ugape_deepen` also stay close across starters.
Shortlist after this section: `best_first`, `sequential_halving_beam`, and `ugape_deepen` stay strong. `progressive_widening` is complete and can be judged on starter gap too.

**7. Training histories and validation histories**

Best training shapes: `sequential_halving_beam`, `ugape_deepen`, `best_first`.
Best validation shapes: `sequential_halving_beam`, `ugape_deepen`, `greedy`.
Worst shapes stay with `random`, `shot`, and nearby rejects.
Shortlist after this section: unchanged.

Algorithms that will stay for future experiments:

- `sequential_halving_beam`: keep. Top-band accuracy/composite and small starter gap. Note: look-ahead deepen uses `MAX_DEPTH = 2`.
- `ugape_deepen`: keep. Top-band accuracy/composite on both starters. Note: deepen also uses `MAX_DEPTH = 2`.
- `best_first`: keep. Balanced across accuracy, starter gap, and middle exploration.
- `beam_search`: keep. Strong middle-band survivor with real look-ahead (`MAX_DEPTH = 2`).
- `montecarlo`: keep as the look-ahead reference / baseline, even though it is unstable here.
- `greedy`: keep as the easy-task filter baseline.
- `random`: keep as the weak baseline control.

Algorithms removed from future experiments:

- `progressive_widening`: drop. Accuracy can look high, but action mix is too narrow. Same exploratory risk as other residual-only winners.
- `successive_rejects`: drop. Strong on medium alone, but pooled stability collapses and the action mix is too narrow.
- `shot`: drop. Weak pooled accuracy/stability and narrow exploration.
- `sequential_halving`: drop. Depth-1 only; beaten by its beam hybrid.
- `ugape`: drop. Depth-1 only; beaten by `ugape_deepen`.
- `hierarchical_search`: drop. Finished the grid, but it did not earn a place in the keep shortlist.

Keep set for the next harder-task experiment: `sequential_halving_beam`, `ugape_deepen`, `best_first`, `beam_search`, plus baselines `montecarlo`, `greedy`, and `random`.

## Conclusions

1. The main goal is a stable simulation algorithm that still looks beyond one local step. MCTS can look ahead, but it is not stable enough here.
2. The current task is still easy: greedy is high. This experiment is a filter for the next harder runs, not a final default lock.
3. After the full grid (`130` / `130`), keep `sequential_halving_beam`, `ugape_deepen`, `best_first`, and `beam_search`, with baselines `montecarlo`, `greedy`, and `random`.
4. Remove `progressive_widening`, `successive_rejects`, `shot`, `sequential_halving`, `ugape`, and `hierarchical_search` from the next experiment set. Narrow action mix and/or weak pooled evidence are the rejection reasons.
5. The planned MNIST grid is complete. Re-test only the keep set on a harder task.

## Next experiments

1. Re-test only the keep set on a harder starter / dataset (for example CIFAR), where greedy should stop looking strong.
2. Prefer methods that stay high without depending on one first residual action.
3. Keep MCTS as a look-ahead reference, not as the default, until seed collapses are fixed or reduced.
