[[Simulation Search Improvement Plan]]

Top ten simulation search candidates for GrowingNN. Every method returns one root action under a time budget through the same `get_action` contract as [[MCTS]], `greedy_alg.py`, and `random_alg.py`.

Read [[Simulation Search Common Steps]] first. That page defines arm, node, expand, score, pull, depth, and why Group 1 has no lookahead.

Implemented modules live under `growingnn/simulation/simulation_algorithms/`. Each file exposes `get_action(traced, running_config)` like `greedy_alg.py`. Experiment 005 selects them with `hp["simulation_alg"]`.

Baselines outside this top ten: current [[MCTS]], greedy depth-1, and random.

GrowingNN needs both:

- Need A. Better ranking when many depth-1 scores are noisy or nearly tied
- Need B. Short selective lookahead so multi-step futures can matter

## Group 1. Root ranking only

These methods stay at depth 1. They expand each chosen root action once, then decide how many score pulls each depth-1 child gets. They do not expand depth-1 children into depth 2. If you need add-then-remove, use Group 2 or Group 3.

### 1. Sequential Halving

What it is: a fixed-budget way to pick the best option among many candidates when each grade is noisy and expensive. It comes from best-arm identification in multi-armed bandits (Karnin, Koren, Somekh).

How the tree is used: only the root decision. Arms = all legal root actions from `generate_all_actions`. There is no next depth. “Arm” here is not a path. It is one depth-1 child.

How expand works: for each root action still alive, expand once into a depth-1 node if that node does not exist yet. See [[Simulation Search Common Steps]].

How scoring works: each pull grades that depth-1 node with `SimulationScore`. If an arm is pulled again, grade it again and average. The mean is the arm’s current value.

How the “worse half” is defined: sort living arms by mean score, high to low. Keep the top half. Drop the bottom half. If the count is odd, keep `ceil(n/2)` best arms. Example with 8 root actions and enough time for about 40 pulls total:

1. Round 1. All 8 arms get some pulls (about the same count). Sort by mean. Keep 4. Drop 4.
2. Round 2. The remaining 4 arms get more pulls. Sort. Keep 2. Drop 2.
3. Round 3. The remaining 2 arms get more pulls. Keep 1. That action is returned.

Budget: convert `simulation_time` into a total pull count `B`, or stop when wall time ends between rounds. Classic Sequential Halving spends about `B / (ceil(log2(n)) * n_round)` pulls per living arm in each round, where `n_round` is how many arms are still alive.

Good: simple; fits a fixed time box; every survivor has a stored mean score.
Bad: no depth 2+; a good arm with unlucky early pulls can be dropped in round 1.

```text
arms = generate_all_actions(root)          # n candidates
for each arm: expand once to a depth-1 node
B = how many SimulationScore calls fit in simulation_time
rounds = ceil(log2(n))
while |arms| > 1:
    pulls_each = max(1, floor(B / (rounds * |arms|)))
    for each arm in arms:
        repeat pulls_each:
            score(arm.depth1_node)         # common score step
            update arm.mean
    sort arms by mean descending
    arms = best half of arms               # keep ceil(|arms|/2)
return the last remaining arm.action
```

### 2. UGapE

What it is: gap-based best-arm identification (Gabillon, Ghavamzadeh, Lazaric). It spends new pulls on arms that still threaten the current ranking of first vs second.

How the tree is used: root only. Same arms as Sequential Halving. No depth 2.

How expand works: expand a root action the first time it is selected. Then only rescore that depth-1 node.

How scoring works: same common pull. UGapE also tracks uncertainty or confidence around each mean. The next pull goes to the arm that most reduces the chance of picking the wrong winner.

Good: strong when many depth-1 scores are almost equal.
Bad: no lookahead; needs a clear stop when time ends.

```text
arms = generate_all_actions(root)
expand and pull each arm at least once
while time left:
    best = arm with highest mean
    challenger = arm that looks closest to beating best
    pull the arm whose extra score most shrinks that gap
return current best.action
```

### 3. Successive Rejects

What it is: another fixed-budget root picker (Audibert, Bubeck, Munos). Each round rejects only the current worst arm, not half of them.

How the tree is used: root only. No depth 2.

How expand and score work: same common expand and pull as Sequential Halving.

How reject works: after each round’s pulls, drop exactly one arm, the lowest mean. Near-tied good arms stay longer than in Sequential Halving.

Good: safer under high score noise.
Bad: still depth-1; often needs more rounds than Sequential Halving.

```text
arms = generate_all_actions(root)
expand each arm to depth 1
for round = 1 .. n-1:
    pull each living arm on the Successive Rejects schedule
    remove the arm with the worst mean
return the last arm.action
```

## Group 2. Selective lookahead

These methods call expand again on depth-1 nodes. That is how they reach depth 2 and can value futures such as add then remove. Child choice is by score or by Sequential Halving, not by random rollout.

### 4. Beam Search

What it is: classical beam search on the action tree. Keep only the best `k` nodes at the current frontier.

How next node works: take every node in the beam, generate its actions, expand each action into a child, score the child, then keep only the top `k` children as the next beam.

How scoring works: each new child is scored once with the common score step. The path remembers its root action.

Return rule: the root action of the best scored node seen, or of the best node in the final beam.

Good: easy to follow; true multi-step lookahead; no random child picks.
Bad: if `k` is small, a weak immediate grade can kill a root action that would look good after one more step.

```text
frontier = []
for a in generate_all_actions(root):
    child = expand(root, a)
    child.score = score(child)
    frontier.append(child)
beam = top_k(frontier)
best = argmax(frontier)
while time left and depth(beam) < d:
    nxt = []
    for node in beam:
        for a in generate_all_actions(node):
            child = expand(node, a)        # go to next depth
            child.score = score(child)
            nxt.append(child)
    beam = top_k(nxt)
    best = better(best, best_in(beam))
return best.root_action
```

### 5. Best-first Search

What it is: a priority-queue search. Always expand the open node with the best score so far.

How next node works: pop the best open node. Generate its actions. Expand and score each child. Push children into the open queue. Optional depth penalty: prefer shallower nodes when scores are close.

How scoring works: common score on each new child. Cap max depth and max nodes.

Good: leftover time goes where grades look best.
Bad: can dig deep into one noisy branch if there is no depth cap.

```text
open = empty max-priority queue
for a in generate_all_actions(root):
    child = expand(root, a); child.score = score(child); push(open, child)
best = best in open
while time left and open not empty:
    node = pop_best(open)
    if node.depth >= max_depth: continue
    for a in generate_all_actions(node):
        child = expand(node, a)            # next node
        child.score = score(child)
        push(open, child)
        best = better(best, child)
return best.root_action
```

### 6. SHOT (Sequential Halving on Trees)

What it is: Sequential Halving used at every node of a depth-limited tree (Cazenave). It is Sequential Halving with real next-node expansion.

How next node works: at a node, the arms are that node’s legal child actions. To evaluate a child arm under a budget share, SHOT expands that child and runs SHOT again on the child with smaller depth left. So depth grows by recursion, not by random rollout.

How scoring works: at max depth, a pull is a normal `SimulationScore` of that leaf node. Above the leaf, a child’s value is the result returned by the recursive SHOT call.

How half-drop works: same as Sequential Halving, but at every node: sort that node’s children by value, keep the better half, spend more budget on survivors.

Good: Need A and Need B in one method; natural replacement for UCT-style [[MCTS]].
Bad: heavier to implement; opening all children at every node is costly without progressive widening.

```text
function shot(node, budget, depth_left):
    if depth_left == 0:
        return score(node)                 # common leaf pull
    arms = generate_all_actions(node)
    for each arm: child[arm] = expand(node, arm)
    while |arms| > 1 and budget left:
        share = budget split by Sequential Halving over |arms|
        for each arm in arms:
            value[arm] = shot(child[arm], share, depth_left - 1)
        arms = better half by value
    return value of best remaining arm

best_root_child = argmax over root children after shot(root, B, d)
return that child’s action
```

## Group 3. Hybrids and scaling

These combine Group 1 ranking with Group 2 expand, or cut the action list before expensive expand and score.

### 7. Sequential Halving then Beam

What it is: Phase 1 is Sequential Halving at the root. Phase 2 is Beam Search that starts only from surviving depth-1 nodes.

How next node works: Phase 1 never goes past depth 1. Phase 2 expands survivors to depth 2, 3, … with the beam rule.

How scoring works: Phase 1 uses repeated pulls on depth-1 nodes. Phase 2 uses one score per new deeper child, same common score step.

Good: clear base case first; futures only for actions that already look good.
Bad: if Phase 1 drops the true best root action, Phase 2 cannot bring it back.

```text
survivors = SequentialHalving(root)        # algorithm 1
beam = survivor depth-1 nodes
best = best survivor
while time left and depth < d:
    beam = top_k(expand and score all actions of beam nodes)
    best = better(best, best_in(beam))
return best.root_action
```

### 8. UGapE then limited deepen

What it is: Phase 1 is UGapE on root arms until time split or the top gap is clear. Phase 2 deepens only the current best arm and its close rivals with beam or best-first.

How next node works: Phase 2 expands those few depth-1 nodes using the Group 2 expand rule.

How scoring works: Phase 1 repeated root pulls; Phase 2 common scores on deeper children.

Good: strong when root scores are nearly tied, then still allows multi-step futures.
Bad: two stop rules to set.

```text
best, rivals = UGapE(root, budget1)
beam = depth-1 nodes of best and rivals
while time left:
    expand and score from beam             # Group 2 next-node step
    shrink beam to top_k
return best.root_action on best path
```

### 9. Progressive Widening plus Halving or Beam

What it is: a scaling wrapper. Do not expand every legal action on the first visit. Unlock more actions as time or visits grow. Then run Sequential Halving or Beam on the open set.

How next node works: only open actions may be expanded. When a new action unlocks, expand it into a new child and score it. Deeper search uses Halving or Beam on open children only.

How scoring works: common score on each expanded child.

Good: needed when `|generate_all_actions(root)|` is huge.
Bad: unlock order matters; a late action may get too little budget.

```text
open_actions = first m root actions in a fixed order
while time left:
    maybe add one more root action into open_actions
    run one Sequential Halving round or one beam deepen step
        using only open_actions for expand
return best open root action
```

### 10. Hierarchical family or layer search

What it is: split root actions into groups by action family or target layer. Pick promising groups first. Run Sequential Halving and a short beam only inside those groups.

How next node works: same common expand, but the legal list is filtered to the chosen groups. Deeper expands can stay inside the group policy or reopen full `generate_all_actions` on children.

How scoring works: short scores to rank groups, then normal pulls inside the winners.

Good: uses structure that current [[MCTS]] ignores.
Bad: a rare good family can lose the group round.

```text
groups = partition(generate_all_actions(root)) by family or layer
rank groups with one cheap or full score sample each
arms = actions from top groups
return SequentialHalvingThenBeam(arms)
```

## What was cut

Not in the top ten for now: F-Race, BRUE, BAST, hill climbing, regularized evolution, filter-then-refine, Gumbel Top-k.

## First implementation wave

All ten are implemented as separate modules:

1. `sequential_halving_alg.py`
2. `ugape_alg.py`
3. `successive_rejects_alg.py`
4. `beam_search_alg.py`
5. `best_first_alg.py`
6. `shot_alg.py`
7. `sequential_halving_beam_alg.py`
8. `ugape_deepen_alg.py`
9. `progressive_widening_alg.py`
10. `hierarchical_search_alg.py`

Experiment driver: `experiments/train_mnist_exp005_simulation_algorithms.py` (fixed Exp 004 `composed_exponential` package; grid over these algs plus montecarlo, greedy, random).

## Comparison with the original growingNN paper

The paper used Monte Carlo simulation because lookahead beat greedy. This top ten keeps expand-and-score lookahead and changes only the policy that chooses which child to expand and how many times to score it.

## Known limitations

These ten are a reasoned shortlist, not measured winners. The live trainer still applies one root action only.
