# Experiment 012: residual-block actions (duplicate and remove)

## Context

This experiment adds actions that duplicate or remove a whole residual block, and asks in four steps whether they help the growing network and whether the search uses them.

## The residual-block action

A residual block is a part of the network whose input is added back to its output: `y = x + F(x)`, usually followed by a ReLU. The action treats each such block as one unit and can **duplicate** it (insert a copy right after it) or **remove** it (connect its input straight to its output).

To find the blocks, the action reads the network as a graph of operations. For every addition node it looks for the latest node from which both branches of the addition start, the block's single entry. Everything between that entry and the addition is the block, if it forms a closed region: nothing inside is used outside the block, and the addition is its only exit. In graph terms these are single-entry, single-exit regions of the computation graph. Blocks that change the tensor shape (downsampling blocks with a projection on the shortcut) are skipped, because a copy would not fit after them. The copy starts as an identity (quasi-identity weights), so the network computes the same function right after the duplication.

![Residual blocks found in a small ResNet, before and after duplicating one](/assets/experiments/012-small-resnet-blocks.png)

> [!CAPTION] Figure 1. A small ResNet built from ResNet-18's blocks, drawn with the project's graph drawer. Left: the blocks the action finds; each colored block can be duplicated or removed, and the gray downsampling block (16 → 32 channels with a projection on the shortcut) is skipped. Right: after duplicating the green block, the copy (also green) is inserted right after it. Produced by running the action itself (`generate_experiment_012_block_diagram.py`).

## Part 1: offering the actions to the search

The search already grew networks one layer at a time. Residual networks suggested another possibility: grow or shrink a whole block, preserving its shortcut connection. We added actions to duplicate and remove residual blocks, then compared runs with those actions enabled against runs without them. This first check kept the small starter, which contained no residual block; duplication would become available only if a suitable block appeared during growth.

### Setup

| Setting | Value |
| --- | --- |
| Datasets, seeds | MNIST, CIFAR-10, CIFAR-100; seeds 1 and 2 per arm |
| Starter | Conv →8, ReLU, pool, Linear (152–1,016 parameters) |
| Schedule | 20 generations × 10 epochs |
| Search | Monte Carlo, 8 s, 1 epoch, 128 simulation examples (old defaults) |
| Arms | Residual-block actions on vs off; all other actions on in both |

> [!CAPTION] Table 1. Setup.

### 1. Comparing runs with the actions on and off

The first comparison is final test accuracy under the same training budget.

![Final test accuracy by seed and arm](/assets/experiments/012-final-test-accuracy.png)

> [!CAPTION] Figure 2. Final test accuracy per seed; horizontal marks are the arm means.

| Dataset | Actions on: mean test (seed range) | Actions off: mean test (seed range) |
| --- | ---: | ---: |
| mnist | 80.62% (63.36–97.88) | 97.50% (97.48–97.52) |
| cifar10 | 51.86% (50.96–52.76) | 59.87% (59.82–59.92) |
| cifar100 | 20.88% (20.68–21.07) | 17.03% (13.85–20.21) |

> [!CAPTION] Table 2. Last recorded test accuracy, two seeds per arm, from the saved snapshot. Ranges are seed minima and maxima, not confidence intervals.

The enabled arm finishes lower on MNIST and CIFAR-10, but higher on CIFAR-100. One enabled MNIST seed ends at only 63.36% test. To interpret these differences, we need to check which actions the search actually applied.

### 2. Checking which actions were selected

![Selected action counts per dataset and arm](/assets/experiments/012-action-counts.png)

> [!CAPTION] Figure 3. Selected actions summed over both seeds (38 per group). No residual-block action was ever selected.

No residual-block action appears among the 228 decisions. The accuracy differences therefore cannot be attributed to applying a duplicate or removal; the other search trajectories also differ.


Script: `experiments/train_cifar10_exp012_0_residual_block_actions.py`.

## Part 2: how a duplicated block should start

Part 1 never applied a residual-block action, and its logs could not explain why. We therefore put a block in the starter and began recording candidate scores. This made it possible to investigate the duplicate itself: copying learned weights changes the network's output, so we compared that initialization with quasi-identity and residual-zero starts, which aim to preserve the output while the new block learns.

### 1. How the search scores each initializer

When a block is duplicated, the new copy needs starting weights. Three ways were compared:

- **Copy:** the new block gets exactly the trained weights of the block it copies. It is immediately a second trained block, but stacking it changes what the network computes, so accuracy usually drops right after the action.
- **Quasi-identity:** every convolution and linear layer in the copy is set to pass its input straight through (1 on the channel diagonal at the kernel centre, 0 elsewhere, biases 0), and BatchNorm statistics are reset. The copy starts close to doing nothing, so the network's output changes only a little, and the block then learns from there.
- **Residual-zero:** a residual block computes `x + F(x)`. The copy's branch `F` gets copied weights, but its last layer is set to zero, so `F(x) = 0` and the new block passes its input through unchanged: the network's output is exactly the same right after insertion. During training the zero layer moves away from zero and the branch gradually switches on, reusing the copied features.

![Residual-block candidate scores by initializer](/assets/experiments/013-block-grades-by-initialization.png)

> [!CAPTION] Figure 4. Simulation scores of residual-block candidates by initializer (triangles: means, orange: medians). Copied weights score clearly lower on both datasets (on CIFAR-10 about 0.29 against about 0.45 for the two identity-style starts). Scores include up to two random rollout actions.

### 2. Following the effect into training

Next we looked at what happened in the runs where the search actually applied a duplicate. Nothing was forced: in each of the 18 runs with block actions, the search was free to pick any action after every generation, and in only 3 runs did it pick a duplicate. All three happened early, after generation 1 or 2, because the starter already contains one residual block, so a duplicate is offered from the first search on. For each of them we compare training and validation accuracy at the end of the generation before the duplicate and at the end of the generation after it.

![Training and validation around the three applied duplicates](/assets/experiments/013-block-duplicate-impact.png)

> [!CAPTION] Figure 5. The three runs in which the search chose a duplicate on its own: CIFAR-10 with quasi-identity (seed 1, after generation 1), CIFAR-10 with residual-zero (seed 1, after generation 1) and CIFAR-100 with residual-zero (seed 1, after generation 2). The vertical line marks the duplicate and the shaded area the following 5-epoch generation. Training accuracy rose afterwards, validation did not; the change also includes ordinary continued training.

Each of these duplicates added the same 2,336 parameters: the initializers only change the starting values of the weights, not their number, and all runs duplicated the same residual block of the same starter. ### 3. Final accuracy: quasi-identity gives the best runs on CIFAR-10

### 3. Final accuracy: quasi-identity gives the best runs on CIFAR-10

Only three applied duplicates is too few to say whether the action helps, so we also compared how the whole runs ended.

Figure 6 shows the final test accuracy of every run, grouped by dataset and by arm: the three initializers with block actions on, and a control group with block actions off. Each dot is one seed and the black mark is the mean of the three seeds. If one initializer made the runs better, its dots would sit clearly above the others.

![Final test accuracy by initializer and dataset](/assets/experiments/013-initialization-test-comparison.png)

> [!CAPTION] Figure 6. Final test accuracy per seed; black marks are means. Quasi-identity leads CIFAR-10 by 0.6 pp, residual-zero leads CIFAR-100; both within seed noise.

| Dataset | Copy | **Quasi-identity** | Residual-zero | Block actions off |
| --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | 61.20% | **62.24%** | 54.66% | 57.40% |
| CIFAR-100 | 23.47% | 25.06% | 25.52% | **26.18%** |

> [!CAPTION] Table 3. Mean final test accuracy over three seeds.

**On CIFAR-10, quasi-identity has the best mean, 4.8 pp above the runs without block actions**, and copy is also above them. On CIFAR-100 all four arms end within about 3 pp and the runs without block actions are slightly ahead. With three seeds and seed spreads of up to 20 pp, these differences are not yet reliable, but quasi-identity is never worse than the other two initializers by more than 0.5 pp, which is why it became the default.


Script: `experiments/train_cifar10_exp012_1_residual_block_action_grading.py`.

## Part 3: longer runs on four datasets — the action is offered but almost never chosen

We then let the search run longer (up to 30 generations of 10 epochs) from the residual starter, with quasi-identity duplicates, on MNIST, CIFAR-10, CIFAR-100 and Tiny ImageNet. The question was simple: given more time and harder datasets, does the search start using the block actions?

![Simulation scores of residual-block candidates](/assets/experiments/014-block-candidate-scores.png)

> [!CAPTION] Figure 7. Simulation scores of every duplicate and remove candidate the search considered, per dataset; bars are means. The actions were scored in every dataset but selected only twice, both times on Tiny ImageNet.

The search did evaluate the block actions in every dataset, but it **never chose one on MNIST, CIFAR-10 or CIFAR-100**, and only twice on Tiny ImageNet. With the action almost never applied, these runs cannot tell whether duplicating a block helps; final accuracy differences between the runs come from other choices of the search. The starter was the same as in Part 2 (one residual block), so the structure is not what changed. The search settings were: Part 3 scored candidates on 128 simulation images instead of 256, with 10-epoch instead of 5-epoch generations. And even in Part 2 the duplicate was picked in only 3 of 18 runs, so seeing none here can also be chance. This is why Part 4 applies the action directly instead of waiting for the search to pick it.

Script: `experiments/train_cifar10_exp012_2_residual_block_multidataset.py`.

## Part 4: testing the duplicate directly

The search rarely picked the duplicate, so we could not tell whether the action is bad or the search just misses it. Part A applies the action by hand, without any search; Part B puts it back into normal growing runs.

### Part A: which version of the duplicate helps most

We took one trained CIFAR-10 network (the residual starter after some training), applied one version of the action to it, trained it for one more generation (10 epochs, 2 seeds) and compared it with the same network trained without any change. **What we check: how much each version of the action improves accuracy over doing nothing.** The versions build on each other:

Each version keeps all the changes of the versions above it, so `+after_relu` means "current, plus side inputs, plus the copy after the ReLU". The names are the labels on the bars in Figure 8.

- **current** (the original action): a block is `x + F(x)` followed by a ReLU. The original duplicate copies the block and inserts the copy *before* that ReLU, so the two blocks are added together without a nonlinearity between them. It only finds blocks whose branch takes input from the block's entry alone. Remove deletes a block completely.
- **+side** (blocks with side inputs): other actions, such as adding a residual convolution, can wire an extra input from an earlier layer into a block's branch. The original action stopped recognising such a block as a block. This version still finds it, so the duplicate can be offered in more networks.
- **+after_relu** (copy placed after the ReLU): the copy is inserted *after* the block's ReLU and has its own ReLU, so the network becomes block → ReLU → copy → ReLU, the same pattern as stacking two blocks in a ResNet.
- **+scale** (gentler start): the copy's branch output is multiplied by a factor ε when it is inserted. With ε = 0 the copy adds nothing at first and the network is unchanged; with ε = 1 the copy starts at full strength.
- **+merge** (softer shrinking): when two identical blocks sit next to each other, shrinking merges them into one block (averaging their weights) instead of deleting one outright.

![Validation gain over unchanged training for each duplicate design](/assets/experiments/015-paired-ladder.png)

> [!CAPTION] Figure 8. Validation gain over the unchanged network after one generation, for each version. Bars are two-seed means; markers are seeds. Left: plain network. Right: a network whose block has an extra input from an earlier layer.

**Answer: inserting the copy after the ReLU works best, +4.2 pp over doing nothing, against +1.9 pp for the original version.** The gain appears only later: right after insertion accuracy dips by about 4.4 pp, recovers after one epoch and pulls ahead by the end of the generation. Removing or merging a block costs 9–13 pp.

![Validation gain at different residual scales](/assets/experiments/015-scale-sweep.png)

> [!CAPTION] Figure 9. Gain over the unchanged network for five values of the scale ε, measured right after insertion, after one epoch and after one generation.

**Answer for the scale: full strength (ε = 1) is best, +4.0 pp after one generation, against +2.8 pp for ε = 0.** So ε = 1 was kept, and the +scale step in Part B is the same as +after_relu.

### Part B: does the search pick the improved duplicate?

Each version from Part A was put back into normal growing runs (CIFAR-10, 30 generations of 10 epochs, seeds 1 and 2, with the older search settings: 8 s and 128 simulation images). **What we check: whether the search now picks the duplicate more often.** We count how often a duplicate was offered, ranked in the top 5, and chosen.

![How often the duplicate is offered, ranked in the top 5, and chosen](/assets/experiments/015-duplicate-selection.png)

> [!CAPTION] Figure 10. Over 58 searches per step: how often a duplicate was offered, ranked in the top 5, and chosen. Side-input support doubled availability but not selection.

Only six block actions were applied in ten enabled growing runs; no merge was applied. More opportunities did not produce a comparable increase in selections. We therefore compare the duplicate's score with unchanged training and with the winning candidate.

![Duplicate score compared with no change and with the winner](/assets/experiments/015-score-gaps.png)

> [!CAPTION] Figure 11. Best duplicate's score minus the no-change score (left) and minus the winner's score (right), per search. The gray band is ±1 standard deviation of score noise. The duplicate scores near no change; selecting the maximum among roughly 150 noisy candidates can favor an unusually high measurement.

| Score comparison | Recorded result |
| --- | --- |
| Early duplicate advantage | About 2 pp |
| Candidate-score noise | About 12 pp standard deviation |

> [!CAPTION] Table 4. The early duplicate advantage is small relative to candidate-score variation. Score noise is not an uncertainty interval for final accuracy.

Side-input support increased availability but not selection. The best final-accuracy step never applied a block action, so its result does not demonstrate the duplicate's benefit. Large score variation is consistent with unreliable selection; it does not prove that every winning candidate won by luck.


Script: `experiments/train_cifar10_exp012_3_duplicate_action_design.py`.

## Summary

The block actions were tested in four steps: offering them to the search (Part 1), choosing how a new block starts (Part 2), longer runs on four datasets (Part 3), and applying the action by hand versus letting the search choose it (Part 4).

![Where the residual-block action ended up](/assets/experiments/012-summary.png)

> [!CAPTION] Final figure. Left: validation gain over the unchanged network after one generation, for each version of the duplicate applied by hand (Part 4 A). Right: mean final CIFAR-10 test accuracy of the growing runs in Part 2, by how the duplicated block starts, and without block actions.

**The action ends in a working final version: the copy is placed after the block's ReLU and starts from quasi-identity weights.** Applied to a trained network it gives +4.2 pp over doing nothing, more than twice the original version (+1.9 pp). In the growing runs of Part 2, the runs that could use quasi-identity duplicates ended with the best mean CIFAR-10 test accuracy, 62.2%, 4.8 pp above the runs without block actions. This version is now the residual-block action in the library. **Room for improvement:** the search still rarely picks the duplicate. In the growing runs of Part 4 it was chosen at most 2 times in 58 searches for any version, and the best version was never chosen (Figure 10). Its early advantage after a short simulation, about 2 pp, is small compared with the noise in candidate scores, about 12 pp. The positive results above come from applying the action by hand and from the few runs where the search did pick it; making the search find it reliably is the next step.

- **Part 1:** with a starter that has no residual block, the search never applied a block action.
- **Part 2:** quasi-identity gives the best runs on CIFAR-10 and became the only initializer.
- **Part 3:** in longer runs on four datasets the search still almost never chose the block actions.
- **Part 4:** applied by hand the duplicate helps (+4.2 pp); inside the search it is rarely chosen.

The action is useful; the search score is not reliable enough to find it. That is the starting point of Experiment 013.
