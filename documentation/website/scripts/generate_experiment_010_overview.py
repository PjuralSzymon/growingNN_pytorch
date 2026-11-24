"""Overview charts for Experiment 010: every one-parameter sweep of both passes in one figure.

The values are the per-candidate seed results (two seeds per candidate, final validation accuracy in %).
They match data/experiments/experiment-010-mcts-meta.json, written by generate_experiment_010_charts.py.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parents[1] / 'app/public/assets/experiments'

# parameter label -> list of (candidate, seed low, seed high, selected)
PASSES = {
    'Pass 1': {
        'UCB1 C': [('0.5', 41.36, 51.88, 0), ('1', 48.24, 64.32, 1), ('2*', 49.62, 50.08, 0),
                   ('4', 40.44, 56.56, 0), ('8', 46.48, 55.02, 0)],
        'UCB1 sqrt': [('no*', 43.20, 57.84, 1), ('yes', 38.38, 42.90, 0)],
        'Propagate rollout': [('no*', 50.94, 61.58, 0), ('yes', 54.20, 60.58, 1)],
        'Rollout depth': [('0', 47.40, 56.20, 1), ('1', 43.78, 49.92, 0), ('2*', 46.00, 52.64, 0),
                          ('3', 34.92, 58.88, 0)],
        'Rollout epochs': [('1*', 35.28, 60.50, 0), ('2', 48.18, 55.02, 1), ('4', 28.02, 41.90, 0)],
        'Rollout LR': [('1e-4*', 24.68, 50.10, 0), ('3e-3', 38.76, 49.42, 0), ('1e-2', 41.36, 54.84, 1)],
        'Min iterations': [('1', 45.68, 47.24, 1), ('3*', 38.62, 42.20, 0), ('5', 37.30, 47.26, 0)],
        'Search time': [('8 s*', 35.66, 48.28, 0), ('16 s', 31.50, 55.90, 1), ('32 s', 34.74, 45.82, 0)],
        'Sim. set': [('0.3%*', 30.88, 47.70, 0), ('5%', 44.62, 53.42, 0), ('10%', 56.34, 57.98, 0),
                     ('30%', 61.66, 61.92, 0), ('40%', 63.74, 64.26, 1)],
    },
    'Pass 2': {
        'UCB1 C': [('0.5', 60.20, 63.66, 0), ('1', 63.24, 64.98, 1), ('2', 57.64, 63.20, 0),
                   ('4', 61.58, 61.62, 0), ('8', 60.14, 64.18, 0)],
        'UCB1 sqrt': [('no', 63.04, 63.82, 1), ('yes', 59.08, 64.76, 0)],
        'Propagate rollout': [('no', 61.30, 64.22, 0), ('yes', 62.44, 63.12, 1)],
        'Rollout depth': [('0', 59.00, 61.98, 0), ('1', 63.10, 65.06, 1), ('2', 58.46, 62.06, 0),
                          ('3', 59.88, 63.22, 0)],
        'Rollout epochs': [('1', 61.32, 64.66, 1), ('2', 54.88, 58.84, 0), ('4', 58.30, 62.04, 0)],
        'Rollout LR': [('1e-4', 61.30, 62.82, 1), ('3e-3', 54.86, 63.24, 0), ('1e-2', 61.70, 62.40, 0)],
        'Min iterations': [('1', 54.44, 63.00, 0), ('3', 57.26, 64.44, 1), ('5', 55.50, 56.08, 0)],
        'Search time': [('8 s', 55.72, 56.76, 0), ('16 s', 56.50, 60.66, 0), ('32 s', 57.06, 64.40, 1)],
        'Sim. set': [('0.3%', 39.20, 54.28, 0), ('5%', 52.34, 53.38, 0), ('10%', 58.24, 60.08, 1),
                     ('30%', 51.68, 56.84, 0), ('40%', 56.84, 59.96, 0)],
    },
}
VARIANTS = [('Phase 1: short training', 58.80, 51.82, 56.51), ('Phase 2: longer training', 59.33, 55.94, 55.53)]


def sweep_overview():
    parameters = list(PASSES['Pass 1'])
    fig, axes = plt.subplots(2, len(parameters), figsize=(18, 6.4), sharey=True, layout='constrained',
                             gridspec_kw={'width_ratios': [len(PASSES['Pass 1'][p]) + 0.6 for p in parameters]})
    for row, (name, sweeps) in enumerate(PASSES.items()):
        for column, parameter in enumerate(parameters):
            axis = axes[row, column]
            for x, (label, low, high, selected) in enumerate(sweeps[parameter]):
                color = 'C1' if selected else 'C0'
                axis.plot([x, x], [low, high], color=color, alpha=.6)
                axis.scatter([x, x], [low, high], s=14, color=color, zorder=3)
                axis.scatter([x], [(low + high) / 2], s=60 if selected else 34, marker='D' if selected else 'o',
                             color=color, zorder=4)
            axis.set_xticks(range(len(sweeps[parameter])), [s[0] for s in sweeps[parameter]], fontsize=8)
            axis.set_xlim(-.6, len(sweeps[parameter]) - .4)
            axis.grid(alpha=.25)
            if row == 0:
                axis.set_title(parameter, fontsize=10, weight='bold')
        axes[row, 0].set_ylabel(f'{name}\nfinal validation (%)')
    axes[0, 0].set_ylim(20, 70)
    fig.suptitle('Experiment 010: one-parameter sweeps (CIFAR-10, 2 seeds per value)', fontsize=13, weight='bold')
    fig.savefig(OUTPUT / '010-sweep-overview.png', dpi=160)
    plt.close(fig)


SLUGS = {'UCB1 C': 'ucb1-c', 'UCB1 sqrt': 'ucb1-sqrt', 'Propagate rollout': 'propagate-rollout',
         'Rollout depth': 'rollout-depth', 'Rollout epochs': 'rollout-epochs', 'Rollout LR': 'rollout-lr',
         'Min iterations': 'min-iterations', 'Search time': 'search-time', 'Sim. set': 'simulation-set'}


def parameter_chart(parameter):
    """Pass 1 and Pass 2 of one parameter side by side: both seeds, their mean, the selected value."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True, layout='constrained')
    for axis, (name, sweeps) in zip(axes, PASSES.items()):
        for x, (label, low, high, selected) in enumerate(sweeps[parameter]):
            color = 'C1' if selected else 'C0'
            axis.plot([x, x], [low, high], color=color, alpha=.6)
            axis.scatter([x, x], [low, high], s=14, color=color, zorder=3)
            axis.scatter([x], [(low + high) / 2], s=60 if selected else 34, marker='D' if selected else 'o',
                         color=color, zorder=4)
        axis.set_xticks(range(len(sweeps[parameter])), [s[0] for s in sweeps[parameter]])
        axis.set_xlim(-.6, len(sweeps[parameter]) - .4)
        axis.set_title(name)
        axis.set_xlabel(parameter)
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Final validation (%)')
    axes[0].set_ylim(20, 70)
    fig.savefig(OUTPUT / f'010-sweep-{SLUGS[parameter]}.png', dpi=160)
    plt.close(fig)


def variants():
    fig, axis = plt.subplots(figsize=(7, 3.6), layout='constrained')
    names = ['Existing Monte Carlo', 'Stabilized MCTS', 'Extended MCTS']
    colors = ['C0', 'C1', 'C2']
    width = .26
    for i, name in enumerate(names):
        values = [row[i + 1] for row in VARIANTS]
        bars = axis.bar([p + (i - 1) * width for p in range(len(VARIANTS))], values, width - .03,
                        color=colors[i], label=name)
        axis.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
    axis.set_xticks(range(len(VARIANTS)), [row[0] for row in VARIANTS])
    axis.set_ylim(40, 65)
    axis.set_ylabel('Mean test accuracy (%)')
    axis.legend(frameon=False, fontsize=8, ncol=3, loc='upper center')
    axis.grid(alpha=.25)
    fig.savefig(OUTPUT / '010-mcts-variants.png', dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sweep_overview()
    for parameter in PASSES['Pass 1']:
        parameter_chart(parameter)
    variants()


if __name__ == '__main__':
    main()
