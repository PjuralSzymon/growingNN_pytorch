# Experiment 018: Monte Carlo 2 and the search score

## Context

Experiment 015 showed that a good action can lose in the search because candidate scores are noisy, and Experiment 017 had one decision where the right choice (A08) made a 20 pp difference but won its search by luck. This experiment uses that decision as a test: if the search is fixed, it should pick A08 every time. It introduces Monte Carlo 2, a cleaned-up version of the search, and improves it step by step. Each fix is then tried in full growth runs, and finally the search is compared with random growth and with the original Monte Carlo search on long runs.

## Summary

- **Question:** why does the search choose weak actions, and what makes it choose good ones?
- **One known-good decision as a test:** at epoch 80 of Experiment 017, adding a convolution with BatchNorm and ReLU (A08) gave +18.6 pp over no change; every other action gave at most +3.7 pp (Figure 1). After several rounds of fixes, Monte Carlo 2 picks A08 in 3 of 3 seeds with a lead of at least 4.6 pp (Figure 3).
- **Better scoring did not give better full-growth runs from the tiny starter:** every run ended at 59–64%, below Experiment 017's 69.5% (Figure 4). On another decision the search's score did not predict real training (Figure 6a).
- **The starter matters:** BatchNorm plus a residual block added about 5 pp after 100 epochs (Figure 7).
- **Over 300 epochs, growth matters most:** search 82.3% test, random growth 72.6% and 85.7%, no growth 53% (Figure 8). Random seed 2 built a 44-times larger network, so the search is not yet shown to beat random growth.

| Setting | Value |
| --- | --- |
| Dataset | CIFAR-10 |
| Search | Monte Carlo 2: every candidate scored at least once, scores are true means, the final choice is the highest mean |
| Phases 1–4, 6, 8, 9, 11 | Replays of the epoch-80 decision of Experiment 017, mostly 3 seeds |
| Phases 5, 7, 10, 12, 14 | Full growth from the 296-parameter starter, 10 generations × 20 epochs, 1–2 seeds |
| Phases 13, 15, 16 | Score against real training, and starter variants |
| Phases 17, 18 | Long runs from a BatchNorm + residual starter (904 parameters), 300–450 epochs |

> [!CAPTION] Table 1. Setup. Full tables and notes for every phase are in the [lab notebook](experiment-018-action-replay-notebook.md).

## Part 1: can the search find the known-good action?

First we measured what each of the five candidate actions of the epoch-80 decision really does, by training each one for a full generation. Then the search was run on the same decision many times while its scoring was changed, to see when it picks the real winner.

![Validation accuracy for each action of the epoch-80 decision](/assets/experiments/018-action-replay-validation.png)

> [!CAPTION] Figure 1. Phase 1: each of the five candidate actions at epoch 80, and no change, trained for 20 epochs from the same checkpoint. Held-out validation accuracy, not used by the search. A08, the convolution with BatchNorm and ReLU, wins by far.

![Simulation accuracy by scoring epochs and action](/assets/experiments/018-scoring-epochs.png)

> [!CAPTION] Figure 2. Phase 2: the simulation score of each action after 1–20 scoring epochs. With 1 epoch A08 looks average; from 5 epochs on it leads.

![A08 lead over the next-best action in each replay](/assets/experiments/018-summary-a08-lead.png)

> [!CAPTION] Figure 3. A08's score minus the best other action's score in four replays, three seeds each; bars are means. A lead ≤ 0 means another action was picked. More scoring epochs (Phase 8) did not help; measuring every candidate several times (Phase 9) and the Phase 11 fixes (build children lazily, restart the scoring learning rate, score futures from trained models) made A08 win every seed.

## Part 2: full growth from the tiny starter

Each improvement from Part 1 was tried in a complete run from the 296-parameter starter of Experiment 010, to check whether a better decision on one checkpoint also gives a better network at the end.

![Final validation of each full-growth run](/assets/experiments/018-summary-growth.png)

> [!CAPTION] Figure 4. Final validation accuracy of every full-growth run from the 296-parameter starter (10 × 20 epochs). Search fixes and settings moved the result by a few points at most, within seed spread; none reached Experiment 017.

![Phase 10 training histories by UCB C](/assets/experiments/018-grid-training-ucb.png)

> [!CAPTION] Figure 5a. Phase 10 training histories for UCB C = 0.05 and 1.0. The models stayed 8 channels wide and stopped improving after generation 2. Training and validation accuracy are about equal, so the networks were too small, not overfitting.

![Phase 14 training history](/assets/experiments/018-widening-training.png)

> [!CAPTION] Figure 5b. Phase 14: widening offered for every convolution. Widening was now chosen 2 times per seed (0 in Phase 12), and seed 1 grew to 51 k parameters, but final validation stayed at 63.7%.

## Part 3: does the score predict real training?

Full-growth runs did not improve, so we checked the score directly: on one decision, every candidate was trained for real and compared with its simulation score. Phase 16 then tested whether the tiny starter itself was holding the runs back.

![Simulation score against real gain for 16 candidates](/assets/experiments/018-summary-phase13.png)

> [!CAPTION] Figure 6a. Phase 13: all 16 growing candidates of the generation-3 decision of a Phase 12 run, each trained for a real 20-epoch generation. The selected candidate (star) had the top score but ranked 9th in real training; score and real gain are unrelated.

![Rank correlation of scoring settings with real gain](/assets/experiments/018-summary-phase15.png)

> [!CAPTION] Figure 6b. Phase 15a: the same 16 candidates re-scored with five settings (CCS simulation set, 3 training seeds). 5 scoring epochs on 10% of the data predict best (0.58) and are the cheapest; longer scoring and averaging three measurements did not help.

![Phase 16 training history](/assets/experiments/018-normalization-training.png)

> [!CAPTION] Figure 7. Phase 16, 100 epochs: tiny starter with a BatchNorm action (57.6%), with BatchNorm in the stem (58.2%), and with BatchNorm and a residual block (63.2%), against Phase 14 (53.8%).

## Part 4: long runs

With the better starter (BatchNorm and a residual block) and the best scoring setting from Phase 15a, the search was run for 300 epochs and compared with growing the network by random actions and with not growing it. Phase 18 then compares the original Monte Carlo search with Monte Carlo 2 on even longer runs.

![Phase 17 final test accuracy and size](/assets/experiments/018-summary-phase17.png)

> [!CAPTION] Figure 8. Phase 17, 300 epochs from the BatchNorm + residual starter: Monte Carlo 2 search, random growth (two seeds) and no growth. Labels give test accuracy and final parameters. The search stopped after 7.5 hours, so it grew the network in only 9 of 14 generations.

![Phase 18 training accuracy, Monte Carlo 1 against Monte Carlo 2](/assets/experiments/018-mc-comparison-train-accuracy.png)

> [!CAPTION] Figure 9. Phase 18 (running): Monte Carlo 1 against Monte Carlo 2 without the search deadline, 15 generations × 30 epochs, training accuracy. Gray lines are generation boundaries; red crosses mark two runs (Monte Carlo 1 seed 1, Monte Carlo 2 seed 2) that crashed with a GPU memory error. So far Monte Carlo 2 seed 1 leads (73.4% validation at epoch 270, 264 k parameters) against Monte Carlo 1 seed 2 (65.7% at epoch 240, 21 k).

## Conclusion

Monte Carlo 2 now finds the known-good action reliably, mainly because it measures every candidate more than once and trains the scoring copies with a restarted learning rate. That did not turn into better full-growth runs from the tiny starter: the networks stayed small and the short score did not predict which action helps most later. Growing clearly beats not growing, and a better starter adds about 5 pp, but whether the search beats random growth is still open because the random runs grew to very different sizes. Phase 18 will show whether Monte Carlo 2 is better than the original search when neither is cut short.

## Limitations

- Most full-growth phases have one or two seeds, while seeds differ by 2–4 pp.
- The score's link to real training is tested on only two decisions (Phases 1 and 13).
- Phase 17 does not control network size.

## Next

- Finish Phase 18 and rerun the two crashed runs.
- Compare search and random growth at equal network size, with three or more seeds.

## Saved results

Script: `experiments/train_cifar10_exp014_action_replay.py` (phases by `--phase`). Runs: `experiments/output/train_cifar10/runs/exp014_action_replay/v1/`. Notebook and phase charts: `generate_experiment_018_report.py`. Summary charts: `generate_experiment_018_summary_charts.py`.
