# From stalled CIFAR-10 training to sustained learning across generations

The original difficulty was not simply a low final score. **GrowingNN could spend 1,000 epochs training and repeatedly searching for structural changes without entering a strong new learning phase.** Removing augmentation made training accuracy look much better, but exposed severe overfitting. The investigation therefore had to explain both weak learning with augmentation and weak generalization without it.

This report follows that investigation through the search experiments, the change in direction toward training and architecture, and the later generation-only studies. It ends with the actual training histories across seeds: an all-action configuration averaging about **90%** and a more consistent configuration finishing above **91% on all three seeds**. These are demonstrated successes, not a claim that every search configuration is now reliable.

**Reading the figures.** “Test” means the official CIFAR-10 test set. Earlier searches sometimes used part of that test set to grade candidates; later runs use a separate validation split. Early and late results therefore establish a development history, not a perfectly controlled causal comparison. Chart dates are the earliest available source-file dates, checked against Git and run records where available. A chart may have been regenerated later; that revision is identified in its caption. Images are preserved with this report.

## 1. What was wrong: much training and search, little late progress

### With augmentation, both curves stalled

An early run used a small convolutional starter, live random crop and horizontal flip, and **10 generations of 100 epochs**. The persistent-tree search was active: nine search calls graded **1,080 candidate actions**. The complete run took about nine hours.

The first generation reached a peak official test accuracy of **57.89%**. Further generations improved it, but progressively less. By generation 7 the peak was **67.29%**; the remaining two generations lifted it only to **67.71%**, a gain of **0.42 percentage points**. Final training accuracy was **66.44%**, and final test accuracy was **67.28%**. The model grew only from **33,390 to 35,964 parameters**.

![One thousand epochs with live augmentation](/assets/experiments/cifar-summary-early-augmented.png)

> [!CAPTION] **Chart date: 5 September 2026.** One seed, 10 × 100 epochs; the run finished the same day. Both training and test flatten near 0.67. Dashed lines mark generation boundaries. The repeated searches did not produce sustained large gains late in training.

The important feature is the shape: architecture changes occurred, yet the curves mostly continued along the same flattening trajectory. This was not an absence of all progress, and it was not a failed search invocation. **The search was doing work, but the selected changes and subsequent training were not producing enough useful learning.**

### Without augmentation, fitting the training set did not solve CIFAR-10

The next run kept the same narrow starter and seed, the same ten-generation length, and the same reported tree settings. Live crop and flip were disabled; simulation training still used augmented images, as it had in the preceding run. The chosen architecture paths then diverged, so this remains a one-seed comparison of the complete evolving process.

Training reached **98.03% at its peak** and ended at **97.95%**. Official test peaked at **54.08% in the first generation**, then declined to **44.59%**. Another thousand epochs made the network better at the training images and worse at unseen images.

![One thousand epochs without live augmentation](/assets/experiments/cifar-summary-early-no-augmentation.png)

> [!CAPTION] **Chart date: 5 September 2026.** One seed, 10 × 100 epochs; run finished the same day. Training approaches 0.98 while test ends near 0.45. Live augmentation is off; simulation training remains augmented. The widening train–test gap is the central failure.

This gave us two distinct problems to investigate: **underfitting the changing augmented views**, and **memorizing the fixed training images when those changes were removed**. A useful solution had to improve both learning and generalization, rather than merely raise training accuracy.

### A larger repeated comparison confirmed the split

We then compared augmentation on and off across four action groups and three seeds per group: **24 completed runs**, each 400 epochs long. The starting architecture was larger than in the early tree runs.

| Live crop and flip | Runs | Mean final training | Mean final official test |
| --- | ---: | ---: | ---: |
| On | 12 | 72.56% | **76.96%** |
| Off | 12 | 96.79% | **65.62%** |

![Repeated augmentation and residual-action comparison](/assets/experiments/cifar-summary-augmentation-comparison.png)

> [!CAPTION] **Chart date: 29 September 2026.** Training above, official test below; augmentation on at left and off at right. Gray permits all actions, blue excludes residual additions. Thin lines are seeds and thick lines are means. The table also includes two other action groups.

Turning off residual additions did not remove the problem: without augmentation, that group ended at **98.49% train and 64.52% test**. With augmentation, its test mean was **77.17%**. Later checks on thinner starters also found memorization without residual additions. Residual growth was therefore not the sole explanation for overfitting.

Training accuracy under augmentation measures harder, changing inputs, while official test uses clean images. A lower training curve alone was not evidence that augmentation was harmful. Across this repeated comparison, live augmentation improved final test by **11.34 points**, even though the entire augmented recipe still fell short of the accuracy we wanted.

## 2. Could different views or better search unlock learning?

During this investigation we explored several plausible explanations. The augmentation schedules and fixed-view runs below preceded the September 29 repeated comparison shown above; the sections group related questions rather than implying each test followed the preceding figure. They were useful diagnostic work, but did not produce a steady sequence of better final models.

| Question | What was observed | What it changed in our understanding |
| --- | --- | --- |
| Would gradually changing augmentation strength help? | Increasing crop/flip probability gave a best completed final test of **68.08%** in the tested schedule grid. Decreasing it let train rise while test finished at **57.46% or 55.79%**. | Easing augmentation could expose memorization again; a schedule alone did not solve the weak learning. One seed per cell, incomplete grid. |
| Would fixed augmented views be easier to learn and still generalize? | Two/four/eight fixed views per original image finished at **62.13% / 67.52% / 70.03%** test. The two-view set reached **100% training accuracy**. | Fixed views could also be memorized. More views helped in these runs, but also meant more examples and computation per epoch. |
| Was the simulation algorithm the main problem? | A later 50-trial mixed search gave algorithm-group mean best tests of **74.77–75.65%**. Its highest individual trial reached **81.04%**. | Those methods were not clearly separated in that sample. Unequal settings and one seed per combination prevent a universal ranking. |
| Would a different simulation score solve it? | A one-seed accuracy-plus-gain score reached **80.41%**, but the broader search did not establish an advantage over accuracy-only scoring. | One promising result was insufficient to make the scoring rule the explanation. |

### A better growth path existed, but finding it was expensive

An earlier diagnostic check, before the September tree runs, evaluated each legal action with **100 epochs on the full training set**, kept the best candidate, and repeated. It found a sequence of convolution widenings that took test accuracy from **57.89% to 80.28%** after six kept steps. Parameters increased to **534,330**, rather than staying close to the tiny starter.

![Full-training evaluation finds a rising widening path](/assets/experiments/cifar-summary-manual-widening.png)

> [!CAPTION] **Chart date: 3 September 2026.** Diagnostic full-training evaluation of candidate actions. The selected widening path kept improving through six steps and was stopped because it was slow. This check preceded the September 5 tree runs; it is shown here to explain what those cheaper searches were trying to reproduce.

That was an important intermediate result: the small starter was not intrinsically incapable of becoming useful. But cheaper simulation did not consistently select the same widenings. The issue was not just whether an action was called “add neurons”; which layer was widened, by how much, and how it trained afterward all mattered.

The emerging picture was that **short candidate training and long real training could rank actions differently**. More search time could be spent evaluating the same imperfect signal. This justified investigating search, but also raised a more basic question: were the graph and training dynamics themselves suitable for CIFAR-10?

## 3. The turning point: examine training and architecture together

We rebuilt ten grown graphs, reset their weights, and trained them with a conventional 200-epoch procedure and live crop/flip. Their best test scores ranged from **76.62% to 83.64%**, averaging **81.38%**. A CIFAR ResNet-18 comparison reached **95.30%**.

![Standard training of grown networks and comparison models](/assets/experiments/cifar-summary-standard-training-histories.png)

> [!CAPTION] **Chart date: 7 October 2026; last modified 12 October.** Training above and official test below, one seed per architecture, 200 epochs. ResNet-18 keeps improving beyond the smaller grown graphs. Model size, normalization, and initial learning rate differ, so this is a diagnostic comparison rather than an isolated trainer ablation.

The comparison did not prove “only the trainer is wrong” or “only the architecture is wrong.” It showed that the same live augmentation could support much stronger learning with a different graph and recipe. We therefore examined momentum, weight decay, BatchNorm, batch size, and learning-rate behavior around mutations.

Simply transferring a high-learning-rate training recipe into GrowingNN was not enough. On one grown graph, restarting the rate at **0.1 every generation**, with momentum and Nesterov enabled, produced generation-end test values of **76.12%, 72.06%, and 62.66%**. Turning momentum off in that setup gave **83.91%, 85.42%, and 85.89%**. A lower-rate composed schedule gave **80.82%, 83.20%, and 84.10%**.

![Training settings change recovery at generation boundaries](/assets/experiments/cifar-summary-optimizer-curves.png)

> [!CAPTION] **Chart date: 9 October 2026; last modified 12 October.** One seed per setting, three generations of 200 epochs. Dashed lines mark boundaries. Some recipes lose progress when the next generation starts, while others recover and exceed earlier performance. Settings also change the subsequent growth path.

The useful lesson was the interaction: a rate and momentum combination that works for one fixed model can behave differently after a structural change. Removing automatic BatchNorm at the high starting rate left the tested graph near **10%**, but that is evidence about this configuration, not a universal claim. Likewise, the successful later recipe uses momentum **0.9**, so this one-seed result is not a reason to declare momentum harmful.

### Another thousand epochs was not the missing ingredient

The better no-momentum variant was extended to **10 × 200 epochs**. Its peak test rose only from **85.98% to 86.31%**; final test was **85.86%**. Later search alternated shrinking and growing the linear layer rather than creating a new strong improvement.

![Long training eventually plateaus even after trainer improvements](/assets/experiments/cifar-summary-long-training-plateau.png)

> [!CAPTION] **Chart date: 10 October 2026; last modified 12 October.** The blue run completed 2,000 epochs; the cyan comparison was unfinished. The completed run plateaued near 86%, despite further generations. The two curves must not be treated as equally completed runs.

We had moved well beyond the earliest 60%-range results, but time alone still did not finish the job. The next step was a simpler combined change to the starting network, the convolution growth operation, and ordinary training.

## 4. A revised package brought test accuracy close to 90%

The next comparison established a clearer baseline. The original configuration used a small starter without BatchNorm and a low-rate composed schedule. Its three final official tests were **72.79%, 63.37%, and 64.20%**, averaging **66.79%**.

The revised package used a wider BatchNorm starter, sequential convolutions inserted **before global pooling**, BatchNorm and ReLU on the inserted convolutions, a higher learning rate, lower weight decay, and gradient clipping. It also restricted the growth choices and added a training-only finish stage. Momentum was **0.9 in both groups in this comparison**; it was not independently established as the source of this improvement.

| Seed | Original final test | Revised end-of-growth test | Revised best finish checkpoint | Revised final checkpoint |
| --- | ---: | ---: | ---: | ---: |
| 110 | 72.79% | 82.16% | 90.19% | 89.98% |
| 111 | 63.37% | 81.44% | 89.65% | 89.41% |
| 112 | 64.20% | 82.38% | 89.84% | 89.66% |
| Mean | **66.79%** | **81.99%** | **89.89%** | **89.68%** |

![Original and revised training packages](/assets/experiments/cifar-summary-original-vs-improved.png)

> [!CAPTION] **Chart date: 13 October 2026** (source metadata and first Git record). Original final test versus revised best and final finish checkpoints. The first 90% crossing was a peak, not the final value of every seed.

The new package had made a large difference, but this was not a matched training-budget ablation: the original had 80 epochs, while the revised package had 56 growth epochs followed by 120 finish epochs. We therefore still needed to know whether **the generation loop itself could train well**, and whether success extended beyond a single favorable checkpoint.

## 5. The generation-only study: what each follow-up taught us

The first generation-only runs, completed on 14 October, answered the basic question: **all three seeds finished above 91% after 120 epochs, without a finish stage**. Their complete histories are shown at the end of this report. The later phases tested the limits of that success. They were follow-ups after the initial breakthrough, not prerequisite steps in a monotonically improving sequence.

### Broader search can still disrupt an otherwise trainable network

With the improved trainer and starter held fixed, allowing more actions produced a wider spread of outcomes. The sequential-convolution reference averaged **91.21%**; all-action search averaged **85.68%**, growth-only **86.21%**, and sequential/residual additions **83.68%**. The broad all-action run had seeds finishing at **85.70%, 90.40%, and 80.94%**. One seed temporarily fell near chance during training and then recovered.

Increasing the all-action simulation package from **8 seconds / 1 epoch / 128 images** to **100 seconds / 8 epochs / 2,000 images** raised its mean to **87.61%**, but seed 111 worsened from **90.40% to 85.04%**. These three simulation settings changed together.

![Training histories under short and longer search](/assets/experiments/cifar-summary-search-budget-histories.png)

> [!CAPTION] **Chart date: 16 October 2026; last modified 30 October.** Rows: sequential-convolution reference, all actions with short simulation, and all actions with longer simulation. Columns: train, official test, validation. Each color is one seed. The middle row shows a temporary collapse; the bottom row learns more steadily but does not improve every seed's final test.

Compared with the early thousand-epoch plateau, the updated networks were clearly learning. But the action-selection process could still interrupt that learning. **Improving the training package did not make every structural decision good.**

### More realistic candidate training helped consistency, but was expensive

We next trained candidate actions on all **45,000 augmented training images**. A one-epoch candidate grade finished at **88.58%, 88.15%, and 87.57%**, averaging **88.10%**. The two- and four-epoch versions did not finish because the grading cost was too high.

Using only the first **5,000 clean training images** for one simulation epoch was cheaper, but produced **77.47%, 90.35%, and 81.70%**: the strong middle seed hid a poor mean and a **12.88-point** spread. One committed neuron-deletion action cut immediate validation from **58.98% to 23.38%**. A candidate winning its short simulation grade did not guarantee a harmless real mutation.

![Candidate-training data and the resulting real histories](/assets/experiments/cifar-summary-simulation-data-histories.png)

> [!CAPTION] **Chart date: 19 October 2026; last modified 30 October.** Rows: sequential-convolution reference, full augmented candidate training, and one-epoch grading on 5,000 clean images. Columns: train, official test, validation; three seeds each. This is a comparison of several changed simulation conditions, not an isolated augmentation ablation.

Extending the clean simulation grade to four epochs brought the three-seed mean to **87.69%**, with a much smaller **2.43-point spread**. Extending it further did not keep improving the result:

| Simulation epochs on 5,000 clean images | 1 | 2 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean final official test | 83.17% | 81.56% | **87.69%** | 86.06% | 84.51% | 85.47% |

This was evidence for a useful middle ground in candidate evaluation, not evidence that the largest grading budget is always best.

### The amount of simulation data mattered, but not monotonically

With four simulation epochs fixed, we reduced the clean subset. **2,500 images** gave the strongest and most consistent result in this sweep: **88.75%, 88.86%, and 89.41%**, a mean of **89.01%** and only **0.66 points** between seeds. Making the subset very small generally brought back wider seed variation.

| Clean simulation images | 5,000 | 2,500 | 1,250 | 625 | 312 | 156 | 78 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Mean final official test | 85.65% | **89.01%** | 88.24% | 84.48% | 81.03% | 83.62% | 83.20% |

![Simulation set size across seeds](/assets/experiments/cifar-summary-simulation-size-comparison.png)

> [!CAPTION] **Chart date: 23 October 2026; last modified 30 October.** Final official test after 120 real epochs. Colored points/lines show seeds, the dark line the mean. The 2,500-image group is both high and tightly grouped; several tiny-subset groups are much less consistent.

The repeated 5,000-image setting averaged **85.65% here versus 87.69% in the earlier sweep**. Search stops on wall-clock time, so even the same nominal recipe and seeds can evaluate different numbers of candidates and follow different paths. This limits how confidently close rankings can be interpreted. Nevertheless, several all-action runs now repeatedly reached the high 80s: this was much broader evidence of learning than one isolated successful seed.

### Longer training inside each generation was not an automatic improvement

Using the 1,250-image simulation subset, we compared 10, 20, and 30 real training epochs within each of 12 generations. Means were **88.24%, 84.65%, and 87.48%**, respectively. One 30-epoch run reached **90.91%**, but another finished at **82.79%**. Giving every selected structure more time could not reliably compensate for a weaker growth path.

![Real training length per generation](/assets/experiments/cifar-summary-longer-generations.png)

> [!CAPTION] **Chart date: 23 October 2026; last modified 30 October.** Runs completed 23–24 October. Rows use 10, 20, or 30 real epochs per generation; columns are train, official test, validation. Each row has three seeds. The total training duration and cosine schedule duration change together.

These histories also distinguish learning from generalization. In the 30-epoch setting, two seeds ended above **97% training accuracy** but at **90.91% and 88.74% test**. More fitting did not uniformly become more test accuracy. We therefore kept ten real epochs per generation and investigated the number of generations separately; those results form part of the final confirmation below.

## 6. Later comparisons support the training-and-architecture explanation

The broad package had worked. The next comparisons asked which parts were carrying that improvement, using a 15-generation all-action setting.

### The learning-rate schedule and convolution package worked better together

The old learning rate peaked at **0.01** and restarted recovery after an action. The new schedule ran globally from **0.1 to 0.001**, without restarting. The old sequential convolution was inserted **after global pooling on a 1×1 map**. The new convolution package placed it **before pooling on an 8×8 map**, followed by BatchNorm and ReLU.

| Learning rate | New-convolution placement/normalization | Mean final test, five seeds |
| --- | --- | ---: |
| Old | Old | 82.17% |
| New | Old | 83.23% |
| Old | New | 83.33% |
| New | New | **88.04%** |

![Learning-rate and convolution changes separate the training curves](/assets/experiments/cifar-summary-training-convolution-histories.png)

> [!CAPTION] **Chart date: 28 October 2026; last modified 30 October.** Comparison runs completed 26–28 October. Five-seed mean train, official test, and validation histories. Green combines both changes; gray changes neither, blue only learning rate, and orange only the convolution package. Individual-seed results are given below.

This is more informative than a single high score: the combined configuration separates from the others during training and on both held-out measurements. Its advantage over the unchanged control is **5.87 percentage points**. The result supports their combination, while leaving pre-pool placement, BatchNorm, and ReLU inseparable within this experiment.

The five final tests with both changes were **89.08%, 89.10%, 88.93%, 86.60%, and 86.50%**. All were well above the old low baseline, but none crossed 90% in this repeat. The first three average **89.04%**, versus **90.04%** in the earlier nominally matching three-seed generation-count study. The evidence is therefore that the updated all-action configuration learns substantially better, with remaining trajectory variation—not that every repetition guarantees 90%.

### Returning to the minimal starter brought poor learning back

We also replaced the large BatchNorm starter with the original small unnormalized model, while retaining the older learning-rate and convolution settings used by the control. On two seeds, the control finished at **81.36% and 81.57%**; the minimal starter finished at **43.58% and 69.00%**.

![Poor learning returns with the minimal starting package](/assets/experiments/cifar-summary-starter-comparison.png)

> [!CAPTION] **Chart date: 30 October 2026** (source metadata and first Git record). Minimal-starter runs completed 29 October. Rows: large normalized starter and minimal unnormalized starter; columns: train, official test, validation. Two seeds each, under the older learning-rate/convolution setup.

The difference appears in the learning histories, not just their final test values. In one minimal run, an immediate validation drop after a neuron deletion was followed by sustained weak performance. Width, BatchNorm, and the action paths all differ, so this is a comparison of starting packages. It does show that the improved starting point was not an incidental cosmetic change in that setup.

The remaining one-at-a-time checks for the search gate, random erasing, smoothing, clipping, weight decay, and shorter training were unfinished in the available report. Their absence matters: the evidence supports the combined recipe and several important interactions, but it does not assign a separate accuracy gain to each optimizer or regularization setting.

## 7. The confirmation: training now continues across generations and seeds

The strongest conclusion comes from the histories, not from claiming that the latest phase always beats the preceding one. The later runs show both a more general all-action route to about 90% and a particularly consistent sequential-growth reference above 91%.

### All-action GrowingNN reaches about 90% in three runs

With ten real epochs per generation, four simulation epochs on 1,250 clean images, and all actions enabled, we compared four generation counts:

| Generations | Real epochs | Seed 110 | Seed 111 | Seed 112 | Mean final test |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 100 | 81.78% | 80.51% | 79.67% | 80.65% |
| 12 | 120 | 90.12% | 85.85% | 88.76% | 88.24% |
| **15** | **150** | **90.62%** | **90.51%** | **88.99%** | **90.04%** |
| 20 | 200 | 91.50% | 83.08% | 90.46% | 88.35% |

![All-action training across generation counts and three seeds](/assets/experiments/cifar-summary-generation-count-histories.png)

> [!CAPTION] **Chart date: 24 October 2026; last modified 30 October.** The new generation-count runs completed 24–25 October; the 12-generation row reuses the earlier subset study. Rows: 10, 12, 15, 20 generations. Columns: train, official test, validation. Every row shows all three seeds, including the weaker 20-generation seed.

The **15-generation row** is the important confirmation. All three training curves rise strongly, and validation and official test rise with them. Final training accuracies are **92.44%, 95.49%, and 93.66%**; validation is **90.34%, 90.12%, and 89.00%**. Official test lies in the same band. This is not the old pattern of near-perfect training paired with a 45–65% test score.

All three 15-generation runs chose at least one convolution addition. The 10-generation runs chose none; the weak 20-generation seed also chose none. That association helps explain the different histories, but it does not isolate the cause: the generation count changes the schedule duration, total training, and search trajectory together. The result establishes a successful all-action configuration without implying that twenty generations must be better than fifteen.

### The clearest repeatable reference passes 90% before the end

The strongest three-seed reference used sequential convolution additions, the larger BatchNorm starter, live augmentation and random erasing, momentum **0.9**, weight decay **0.0001**, clipping, and label smoothing. Its cosine learning rate ran from **0.1 to 0.001 across 120 epochs**. There was **no separate finish stage**: twelve generations performed the entire training, with eleven structural additions between them.

![Sustained learning in the three successful generation-only runs](/assets/experiments/cifar-summary-final-training-curves.png)

> [!CAPTION] **Chart date: 14 October 2026** (source metadata and first Git record); last modified 30 October. The original breakthrough runs, shown here as the final reference. Left: train; middle: official test; right: validation. All three seeds pass 90% official test by epoch 100 and finish above 91%.

The generation-end measurements make the contrast with the early plateau concrete:

| Real epoch | Seed 110 test | Seed 111 test | Seed 112 test | What the stage shows |
| --- | ---: | ---: | ---: | --- |
| 10 | 59.27% | 58.32% | 63.35% | The starter alone is still modest after the first generation. |
| 20 | 76.77% | 75.48% | 72.70% | The next generation makes a substantial gain on every seed. |
| 40 | 81.65% | 83.80% | 83.74% | Learning continues beyond the early rise. |
| 70 | 88.39% | 88.20% | 88.09% | The seeds converge near 88%; later training remains productive. |
| 100 | 90.55% | 90.82% | 90.64% | All three have crossed 90%, before the last generation. |
| 120 | **91.11%** | **91.17%** | **91.36%** | Every seed ends at its highest measured official test. |

This is not evidence that an untrained added layer instantly creates the gain. Immediate validation changes at the successful insertions were at most **0.08 percentage points**. The **17.50, 17.16, and 9.35-point** gains between epochs 10 and 20 occurred during the subsequent ten epochs of training. The meaningful improvement is that the evolving network **can use the following training to learn more**, rather than merely changing its graph while remaining on a flat trajectory.

| Final measurement | Seed 110 | Seed 111 | Seed 112 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Training | 96.61% | 96.38% | 96.37% | 96.45% |
| Validation | 91.60% | 91.28% | 90.76% | 91.21% |
| Official test | **91.11%** | **91.17%** | **91.36%** | **91.21%** |

These runs used **45,000 training images, 5,000 separate validation images, and the official 10,000-image test set logged separately**. The graph grew from **187,008 to 1,813,248 parameters**. The three final tests are within **0.25 points** of one another, and both validation and test confirm the gain. A roughly five-point train–validation gap remains, but the severe early generalization failure does not.

The evidence does not yet establish the advantage of simulation over predetermined growth or a fixed architecture, and repeated examination of official-test results influenced the research process. It does establish the practical result we were seeking: **GrowingNN went from an augmented run that ended near 67% after 1,000 epochs to generation-only runs that all ended above 91% after 120 epochs, while continuing to improve across generations.** These are different configurations rather than an isolated speed or single-change comparison. The demonstrated change is from weak or memorizing training to sustained CIFAR-10 learning that also appears on unseen images.
