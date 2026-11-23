"""Summary charts for the compact Experiment 018 page.

The values are copied from the tables of the full Experiment 018 lab notebook
(experiment-018-action-replay-notebook.md, written by generate_experiment_018_report.py).
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parents[1] / 'app/public/assets/experiments'
ORANGE, BLUE, GRAY, LIGHT = 'C1', 'C0', 'C2', 'C7'

# A08 score minus the best other action's score (pp), seeds 1-3, in each replay of the epoch-80 decision.
A08_LEADS = [
    ('Phase 6\nfixed MC2, 5 epochs', [6.2, 0.0, 4.2]),
    ('Phase 8\n20 scoring epochs', [5.0, -0.1, 4.5]),
    ('Phase 9\n5 min, one seed per search', [1.6, 6.3, 3.1]),
    ('Phase 11\nlazy expansion, LR restart,\nfutures from trained models', [7.2, 4.6, 6.6]),
]

# Final validation (%) of full-growth runs from the 296-parameter starter, 10 generations x 20 epochs.
GROWTH = [
    ('Phase 5\nMC2, 20 scoring epochs', [64.02]),
    ('Phase 7\nfixed MC2, 5 epochs', [59.84]),
    ('Phase 10\nUCB C x sim. set grid', [60.76, 59.38, 61.92, 63.06]),
    ('Phase 12\nsimulation fixes', [60.90, 63.28]),
    ('Phase 14\nwidening everywhere', [63.68]),
]
EXP017 = 69.48

# Phase 13: one late decision, every growing candidate trained for real against an unchanged control.
PHASE13 = [  # (simulation score, real gain over control in pp, selected)
    (0.4500, 3.08, 0), (0.4714, 2.32, 0), (0.4469, 1.76, 0), (0.5004, 1.58, 0), (0.3629, 1.42, 0),
    (0.4797, 1.18, 0), (0.4627, 1.12, 0), (0.4699, 0.04, 0), (0.5165, -0.38, 1), (0.4103, -0.40, 0),
    (0.5094, -1.88, 0), (0.5096, -2.18, 0), (0.4402, -3.42, 0), (0.4839, -3.68, 0), (0.4379, -4.60, 0),
    (0.4801, -6.30, 0),
]

# Phase 15a: rank correlation between simulation score and real gain, per scoring setting.
PHASE15 = [('5 ep, 10%', 0.58, 0.43), ('10 ep, 10%', 0.51, 0.38), ('5 ep, 25%', 0.40, 0.46),
           ('10 ep, 25%', 0.16, 0.44), ('15 ep, 25%', 0.18, 0.09)]

# Phase 17: 300 epochs from the BatchNorm + residual starter (904 parameters).
PHASE17 = [('Search\n(seed 1)', 82.26, 122_624, ORANGE), ('Random\nseed 1', 72.56, 11_852, GRAY),
           ('Random\nseed 2', 85.65, 5_388_544, GRAY), ('No growth\nseed 1', 53.43, 904, LIGHT),
           ('No growth\nseed 2', 53.22, 904, LIGHT)]


def style(axis):
    axis.grid(alpha=.25)


def a08_leads():
    fig, axis = plt.subplots(figsize=(9, 4.2), layout='constrained')
    for x, (_, leads) in enumerate(A08_LEADS):
        axis.bar(x, sum(leads) / 3, .55, color=LIGHT, alpha=.5)
        for seed, lead in enumerate(leads):
            axis.scatter(x + (seed - 1) * .12, lead, s=40, zorder=3, color=[BLUE, ORANGE, GRAY][seed],
                         label=f'Seed {seed + 1}' if x == 0 else None)
    axis.axhline(0, color='black', lw=.8)
    axis.set_xticks(range(len(A08_LEADS)), [name for name, _ in A08_LEADS], fontsize=8)
    axis.set_ylabel('A08 lead over next-best action (pp)')
    axis.set_title('Does the search pick the action that really wins (A08)?', fontsize=11, weight='bold')
    axis.legend(frameon=False, fontsize=8)
    style(axis)
    fig.savefig(OUTPUT / '018-summary-a08-lead.png', dpi=160)
    plt.close(fig)


def growth():
    fig, axis = plt.subplots(figsize=(9, 4.2), layout='constrained')
    for x, (_, values) in enumerate(GROWTH):
        axis.scatter([x] * len(values), values, s=50, color=BLUE, zorder=3)
    axis.axhline(EXP017, color=ORANGE, ls='--', lw=1.2)
    axis.text(len(GROWTH) - .5, EXP017 + .4, 'Experiment 017 (69.5%)', color=ORANGE, ha='right', fontsize=8)
    axis.set_xticks(range(len(GROWTH)), [name for name, _ in GROWTH], fontsize=8)
    axis.set_xlim(-.5, len(GROWTH) - .5)
    axis.set_ylim(55, 72)
    axis.set_ylabel('Final validation accuracy (%)')
    axis.set_title('Full growth from the 296-parameter starter: one dot per run', fontsize=11, weight='bold')
    style(axis)
    fig.savefig(OUTPUT / '018-summary-growth.png', dpi=160)
    plt.close(fig)


def phase13():
    fig, axis = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
    for score, gain, selected in PHASE13:
        axis.scatter(100 * score, gain, s=90 if selected else 40, marker='*' if selected else 'o',
                     color=ORANGE if selected else BLUE, zorder=3)
    axis.annotate('selected', (100 * .5165, -.38), (49.5, -3), fontsize=8, color=ORANGE,
                  arrowprops={'arrowstyle': '->', 'color': ORANGE})
    axis.axhline(0, color='black', lw=.8)
    axis.set_xlabel('Simulation score (%)')
    axis.set_ylabel('Real gain over unchanged model (pp)')
    axis.set_title('Phase 13: simulation score vs real training\n(16 candidates, rank correlation -0.15)',
                   fontsize=10, weight='bold')
    style(axis)
    fig.savefig(OUTPUT / '018-summary-phase13.png', dpi=160)
    plt.close(fig)


def phase15():
    fig, axis = plt.subplots(figsize=(7, 3.8), layout='constrained')
    width = .36
    for i, (label, color) in enumerate([('1 measurement', BLUE), ('Mean of 3', ORANGE)]):
        bars = axis.bar([x + (i - .5) * width for x in range(len(PHASE15))], [row[i + 1] for row in PHASE15],
                        width - .03, color=color, label=label)
        axis.bar_label(bars, fmt='%.2f', fontsize=7, padding=2)
    axis.set_xticks(range(len(PHASE15)), [row[0] for row in PHASE15], fontsize=8)
    axis.set_ylabel('Rank correlation with real gain')
    axis.set_ylim(0, .7)
    axis.set_title('Phase 15a: which scoring setting predicts real training', fontsize=10, weight='bold')
    axis.legend(frameon=False, fontsize=8)
    style(axis)
    fig.savefig(OUTPUT / '018-summary-phase15.png', dpi=160)
    plt.close(fig)


def phase17():
    fig, axis = plt.subplots(figsize=(7, 3.8), layout='constrained')
    bars = axis.bar(range(len(PHASE17)), [row[1] for row in PHASE17], .6, color=[row[3] for row in PHASE17])
    axis.bar_label(bars, [f'{row[1]:.1f}%\n{row[2]:,} p.' for row in PHASE17], fontsize=7, padding=2)
    axis.set_xticks(range(len(PHASE17)), [row[0] for row in PHASE17], fontsize=8)
    axis.set_ylim(0, 100)
    axis.set_ylabel('Test accuracy (%)')
    axis.set_title('Phase 17: search vs random growth, 300 epochs', fontsize=10, weight='bold')
    style(axis)
    fig.savefig(OUTPUT / '018-summary-phase17.png', dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    a08_leads()
    growth()
    phase13()
    phase15()
    phase17()


if __name__ == '__main__':
    main()
