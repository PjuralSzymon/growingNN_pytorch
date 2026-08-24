[[Simulation Search Improvement Plan]]

Keep-set simulation search methods after Experiment 005. Every method returns one root action under a time budget through the same `get_action` contract as [[MCTS]], `greedy_alg.py`, and `random_alg.py`.

Read [[Simulation Search Common Steps]] first. That page defines arm, node, expand, score, rescore, depth, and why some methods stay at depth 1.

Implemented modules live under `growingnn/simulation/simulation_algorithms/`. Each file exposes `get_action(traced, running_config)` like `greedy_alg.py`. Experiment drivers select them with `hp["simulation_alg"]`.

Baselines kept for future grids: current [[MCTS]], greedy depth-1, and random.

Root first-pass rule for the four keep-set look-ahead modules below (`beam_search_alg.py`, `best_first_alg.py`, `sequential_halving_beam_alg.py`, `ugape_deepen_alg.py`): each grades every legal root action once in its own first pass before any timed cut, rival pull, or deepen. That first pass may overrun `simulation_time`. After every root arm has a grade, the algorithm continues with its normal exploration under the remaining budget. Greedy and random do not follow this rule. There is no shared cover helper; each file keeps its own loop.

GrowingNN needs both:

- Need A. Better ranking when many depth-1 scores are noisy or nearly tied
- Need B. Short selective lookahead so multi-step futures can matter

## Group 1. Selective lookahead

These methods call expand again on depth-1 nodes. That is how they reach depth 2 and can value futures such as add then remove.

### 1. Beam Search

What it is: classical beam search on the action tree. Keep only the best `k` nodes at the current frontier. Module: `beam_search_alg.py`.

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
            child = expand(node, a)
            child.score = score(child)
            nxt.append(child)
    beam = top_k(nxt)
    best = better(best, best_in(beam))
return best.root_action
```

### 2. Best-first Search

What it is: a priority-queue search. Always expand the open node with the best score so far. Module: `best_first_alg.py`.

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
        child = expand(node, a)
        child.score = score(child)
        push(open, child)
        best = better(best, child)
return best.root_action
```

## Group 2. Root ranking then deepen hybrids

These combine depth-1 ranking with selective expand. `MAX_DEPTH = 2` in the keep-set modules.

### 3. Sequential Halving then Beam

What it is: Phase 1 is Sequential Halving at the root. Phase 2 is Beam Search that starts only from surviving depth-1 nodes. Module: `sequential_halving_beam_alg.py`.

How next node works: Phase 1 never goes past depth 1. Phase 2 expands survivors to depth 2 with the beam rule.

How scoring works: first grade every root arm once, then Phase 1 rescores and halves under the root time split. Phase 2 uses one score per new deeper child.

Good: clear base case first; futures only for actions that already look good.
Bad: if Phase 1 drops the true best root action, Phase 2 cannot bring it back.

```text
for each root action: expand once; score once
survivors = SequentialHalving(rescored root arms, root time budget)
beam = survivor depth-1 nodes
best = best survivor
while time left and depth < d:
    beam = top_k(expand and score all actions of beam nodes)
    best = better(best, best_in(beam))
return best.root_action
```

### 4. UGapE then limited deepen

What it is: Phase 1 is UGapE on root arms until time split or the top gap is clear. Phase 2 deepens only the current best arm and its close rivals with beam. Module: `ugape_deepen_alg.py`.

How next node works: Phase 2 expands those few depth-1 nodes using the Group 1 expand rule.

How scoring works: first pull every root arm once, then Phase 1 UGapE rival pulls under the root time split; Phase 2 common scores on deeper children.

Good: strong when root scores are nearly tied, then still allows multi-step futures.
Bad: two stop rules to set.

```text
for each root arm: pull once
best, rivals = UGapE(root, remaining root budget)
beam = depth-1 nodes of best and rivals
while time left:
    expand and score from beam
    shrink beam to top_k
return best.root_action on best path
```

## What Exp 005 removed

Modules deleted after the MNIST grid. Historical run folders and the website Experiment 005 page still document them.

- `sequential_halving_alg.py` — depth-1 only; beaten by `sequential_halving_beam_alg.py`
- `ugape_alg.py` — depth-1 only; beaten by `ugape_deepen_alg.py`
- `successive_rejects_alg.py` — strong on medium alone; pooled stability and action mix too weak
- `shot_alg.py` — weak pooled accuracy/stability and narrow exploration
- `progressive_widening_alg.py` — accuracy can look high; action mix too narrow
- `hierarchical_search_alg.py` — finished the grid; did not earn a keep place

Also cut earlier and never shipped in code: F-Race, BRUE, BAST, hill climbing, regularized evolution, filter-then-refine, Gumbel Top-k.

## Keep-set modules

1. `beam_search_alg.py`
2. `best_first_alg.py`
3. `sequential_halving_beam_alg.py`
4. `ugape_deepen_alg.py`

Plus baselines: `montecarlo_alg.py`, `greedy_alg.py`, `random_alg.py`.

Experiment 005 driver keep set: `experiments/train_mnist_exp005_simulation_algorithms.py`.

## Comparison with the original growingNN paper

The paper used Monte Carlo simulation because lookahead beat greedy. The keep set keeps expand-and-score lookahead and changes only the policy that chooses which child to expand and how many times to score it.

## Known limitations

The live trainer still applies one root action only. MNIST evidence is for the Exp 005 package, not a harder task yet.
