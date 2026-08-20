[[Simulation Search Improvement Plan]]
[[Candidate Simulation Algorithms]]

All candidate search methods share the same building blocks. Only the policy that chooses which node to expand next changes. This page defines those shared steps so [[Candidate Simulation Algorithms]] can stay short and precise.

## Words we use

Root state: the live model copy at the start of `get_action` in the current generation.

Action: one legal architecture mutation from `generate_all_actions` in `growingnn/actions/registry.py`. Example: add a residual conv between two layer ids, or delete one layer.

Arm: one candidate choice at a decision point. At the root, each arm is one root action. Deeper in the tree, each arm is one child action from the current node. Arm does not mean a deeper path by itself. It means one edge out of the current node.

Node: one concrete model state in the search tree. The root node is the start state. A child node is the model after one action was executed on a copy.

Path: a sequence of actions from the root. GrowingNN still returns only the first action of the chosen path to the live trainer. Deeper actions exist only to estimate whether that first action was good.

Budget: wall-clock time from `simulation_scheduler.simulation_time`, or an equivalent count of score calls that fit in that time.

Pull: one full grade of one node. A pull always means: take that node’s model, run [[Scoring function]] through `run_simulation_scoring_gradient_descent`, store one number. Repeating a pull on the same logical arm means grading that same post-action state again, or rebuilding and grading it again, to reduce noise.

Mean score of an arm: average of all pulls done for that arm so far.

## Common step 1. Generate actions

From a node’s model, call `generate_all_actions(traced, config)`.

That list is the full child set of the node. Group 1 methods use only the root list. Lookahead methods may call this again on child nodes.

## Common step 2. Expand one action into a child node

Expand is the same for every algorithm:

1. Deep-copy the parent node’s traced model.
2. Execute one chosen action on the copy (`action.execute`).
3. Optionally run a short train on the copy if the search config asks for it.
4. Store the child node: parent link, action that created it, new model, depth = parent depth + 1, and the root action of this path.

Expand does not grade. Expand only builds the next node.

Going to the next node always means: pick one unused or selected action from the parent, expand it, then later score that child.

## Common step 3. Score a node

Score is also shared:

1. Take the child node’s model.
2. Call `SimulationScore.score` / `run_simulation_scoring_gradient_descent` on the simulation loaders from [[Simulation Set]].
3. Save the number on the node and add it into the arm’s pull history.

A depth-1 method only scores nodes created by one root expand. A lookahead method may expand a child again and score grandchildren.

## Common step 4. Choose what to return

When time ends, every method returns one root action: the first edge of the best path found.

It does not return the full path to the live model. `train_generations` in `growingnn/training/trainer.py` still executes only that one action.

## How depth works

Depth 0: root node, current model.

Depth 1: after one expand from the root. This is the set of live candidates.

Depth 2+: after more expands. Used only to estimate the value of the depth-1 action that started the path.

Example of a multi-step idea: depth 1 adds a layer, depth 2 removes another layer. The search may like that path, but the live model only receives the depth-1 add. The depth-2 remove is evidence, not a live plan.

## Why Group 1 methods exist

Group 1 methods never call expand on depth-1 children. They only:

1. generate root actions
2. expand each root action once into a depth-1 node
3. pull scores on those depth-1 nodes, maybe several times
4. pick the best root action

They are in the top ten because noisy root ranking is a real failure mode of current [[MCTS]], and because hybrids reuse them as Phase 1. They cannot see add-then-remove by themselves. That is why Group 2 and Group 3 exist.

## Comparison with the original growingNN paper

The paper already used generate, execute on a copy, short train, and grade. This page only names those steps so new algorithms share one expand and score path.

## Known limitations

Exact pull cost depends on `simulation_epochs` and which terms are enabled in `SimulationScore`. Budget conversion from seconds to pull count is an implementation detail of each experiment.
