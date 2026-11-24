"""Summary charts for the compact Experiment 018 page.

The values are copied from the tables of the full Experiment 018 lab notebook
(experiment-018-action-replay-notebook.md, written by generate_experiment_013_report.py).
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

OUTPUT = Path(__file__).resolve().parents[1] / 'app/public/assets/experiments'
RUN = Path(__file__).resolve().parents[3] / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1'
PHASE18_EPOCHS, PHASE18_GENERATIONS = 30, 10  # the report shows the first 300 epochs (10 generations)
PHASE18_CRASHED = {(1, 1), (2, 2)}  # (Monte Carlo version, seed) stopped by a CUDA error
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


def _phase18_test(version, seed):
    """Test accuracy after each generation, from the run log (the board stores only train/validation)."""
    log = RUN / f'mc_comparison_mc{version}_seed{seed}.log'
    if not log.is_file():
        return []
    pattern = re.compile(r'Generation (\d+) test_acc ([\d.]+)')
    return [(int(g), 100 * float(a)) for g, a in pattern.findall(log.read_text(encoding='utf-8', errors='replace'))
            if int(g) < PHASE18_GENERATIONS]


def phase18():
    """Phase 18 training accuracy, test accuracy after each generation and parameters, per run."""
    if not (RUN / 'mc_comparison_mc1').is_dir():
        return
    fig, (train, test, size) = plt.subplots(3, 1, figsize=(10, 13.5), sharex=True, layout='constrained',
                                            gridspec_kw={'height_ratios': [2, 2, 1.3]})
    for version, color in ((1, 'C0'), (2, 'C1')):
        for seed, style in ((1, '-'), (2, ':')):
            metrics = RUN / f'mc_comparison_mc{version}/seed_{seed}/board/metrics/training.json'
            if not metrics.is_file():
                continue
            epochs = [row for row in json.loads(metrics.read_text(encoding='utf-8'))['epochs']
                      if row['globalEpoch'] < PHASE18_EPOCHS * PHASE18_GENERATIONS]
            x = [row['globalEpoch'] + 1 for row in epochs]
            crashed = (version, seed) in PHASE18_CRASHED
            label = f'Monte Carlo {version}, seed {seed}' + (' (crashed)' if crashed else '')
            train.plot(x, [100 * row['trainAcc'] for row in epochs], color=color, ls=style, label=label)
            size.plot(x, [row['paramCount'] / 1000 for row in epochs], color=color, ls=style, drawstyle='steps-post')
            points = _phase18_test(version, seed)
            test.plot([PHASE18_EPOCHS * (g + 1) for g, _ in points], [a for _, a in points], color=color,
                      ls=style, marker='o', ms=4, label=label)
            if crashed:
                for axis, y in ((train, 100 * epochs[-1]['trainAcc']), (size, epochs[-1]['paramCount'] / 1000)):
                    axis.plot(x[-1], y, marker='X', color='red', ms=9, zorder=5)
    total = PHASE18_EPOCHS * PHASE18_GENERATIONS
    for axis in (train, test, size):
        axis.set_xlim(0, total)
        axis.xaxis.set_major_locator(MultipleLocator(PHASE18_EPOCHS))
        for boundary in range(PHASE18_EPOCHS, total, PHASE18_EPOCHS):
            axis.axvline(boundary + .5, color='gray', alpha=.4, lw=.8, zorder=0)
        axis.grid(alpha=.25, axis='y')
    top = train.secondary_xaxis('top')
    top.set_xticks([PHASE18_EPOCHS * (g + .5) for g in range(PHASE18_GENERATIONS)],
                   [str(g) for g in range(PHASE18_GENERATIONS)])
    top.tick_params(length=0, labelsize=8)
    top.set_xlabel('Generation (30 epochs each)')
    train.set_ylabel('Training accuracy (%)')
    test.set_ylabel('Test accuracy (%)')
    size.set_ylabel('Parameters (thousands)')
    size.set_ylim(bottom=0)
    size.set_xlabel('Training epoch')
    train.legend(fontsize=8, loc='lower right')
    fig.suptitle('Phase 18: Monte Carlo 1 against Monte Carlo 2', weight='bold')
    fig.savefig(OUTPUT / '018-mc-comparison-train-test-params.png', dpi=160)
    plt.close(fig)


def final_summary():
    """Left: A08's lead in each replay of the epoch-80 decision (mean of three seeds).
    Right, stacked: Phase 18 training accuracy per epoch, test accuracy after each generation and
    parameters, Monte Carlo 1 against Monte Carlo 2 (the surviving seed of each)."""
    fig = plt.figure(figsize=(13, 10.5), layout='constrained')
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.5], height_ratios=[1.3, 1, .8])
    left = fig.add_subplot(grid[:, 0])
    names = [name.split(chr(10))[0] for name, _ in A08_LEADS]
    means = [sum(leads) / len(leads) for _, leads in A08_LEADS]
    wins = [sum(lead > 0 for lead in leads) for _, leads in A08_LEADS]
    bars = left.bar(names, means, color=['C7'] * (len(means) - 1) + ['C2'])
    left.bar_label(bars, [f'+{m:.1f} pp' + chr(10) + f'ahead in {w}/3 seeds' for m, w in zip(means, wins)], fontsize=8)
    left.set_ylim(0, 8)
    left.set_ylabel('A08 lead over next-best action (pp)')
    left.set_title('Replays: the fixed search finds the known-good action', fontsize=11)
    left.grid(alpha=.25, axis='y')

    train = fig.add_subplot(grid[0, 1])
    test = fig.add_subplot(grid[1, 1], sharex=train)
    size = fig.add_subplot(grid[2, 1], sharex=train)
    for version, seed, color in ((1, 2, 'C0'), (2, 1, 'C1')):
        epochs = [row for row in json.loads((RUN / f'mc_comparison_mc{version}/seed_{seed}/board/metrics/training.json')
                                            .read_text(encoding='utf-8'))['epochs']
                  if row['globalEpoch'] < PHASE18_EPOCHS * PHASE18_GENERATIONS]
        x = [row['globalEpoch'] + 1 for row in epochs]
        label = f'Monte Carlo {version} (seed {seed})'
        train.plot(x, [100 * row['trainAcc'] for row in epochs], color=color, label=label)
        size.plot(x, [row['paramCount'] / 1000 for row in epochs], color=color, drawstyle='steps-post')
        points = _phase18_test(version, seed)
        test.plot([PHASE18_EPOCHS * (g + 1) for g, _ in points], [a for _, a in points], color=color, marker='o', ms=4)
    total = PHASE18_EPOCHS * PHASE18_GENERATIONS
    for axis in (train, test, size):
        axis.set_xlim(0, total)
        axis.xaxis.set_major_locator(MultipleLocator(PHASE18_EPOCHS * 2))
        for boundary in range(PHASE18_EPOCHS, total, PHASE18_EPOCHS):
            axis.axvline(boundary + .5, color='gray', alpha=.4, lw=.8, zorder=0)
        axis.grid(alpha=.25, axis='y')
    train.set_title('Phase 18 long run: Monte Carlo 2 ahead of Monte Carlo 1', fontsize=11)
    train.set_ylabel('Training acc. (%)')
    test.set_ylabel('Test acc. (%)')
    size.set_ylabel('Parameters (k)')
    size.set_ylim(bottom=0)
    size.set_xlabel('Training epoch (gray lines: generations of 30 epochs)')
    train.legend(fontsize=8, loc='lower right')
    fig.savefig(OUTPUT / '013-summary.png', dpi=160)
    plt.close(fig)


# Phase 16: validation after 100 epochs for each starter, the same Monte Carlo 2 search.
PHASE16 = [('Tiny starter' + chr(10) + '(Phase 14)', 53.78), ('Tiny starter +' + chr(10) + 'BatchNorm action', 57.64),
           ('BatchNorm' + chr(10) + 'in the stem', 58.24), ('BatchNorm +' + chr(10) + 'residual block', 63.24)]


def phase16():
    fig, axis = plt.subplots(figsize=(8, 4), layout='constrained')
    bars = axis.bar([name for name, _ in PHASE16], [value for _, value in PHASE16],
                    color=['C7', 'C7', 'C7', 'C2'])
    axis.bar_label(bars, [f'{value:.1f}%' for _, value in PHASE16])
    axis.set_ylim(45, 68)
    axis.set_ylabel('Validation accuracy at epoch 100 (%)')
    axis.set_title('Phase 16: which starting network trains best', fontsize=11)
    style(axis)
    fig.savefig(OUTPUT / '013-summary-phase16.png', dpi=160)
    plt.close(fig)


def _chain(axis, y, steps, title):
    """One row of boxes joined by arrows. steps: (text, colour) pairs."""
    from matplotlib.patches import FancyBboxPatch
    axis.text(0, y + .62, title, fontsize=10, weight='bold', va='bottom')
    for x, (text, color) in enumerate(steps):
        axis.add_patch(FancyBboxPatch((x * 1.25, y), 1.05, .5, boxstyle='round,pad=.03',
                                      facecolor=color, edgecolor='black', lw=.8))
        axis.text(x * 1.25 + .525, y + .25, text, ha='center', va='center', fontsize=7.5)
        if x:
            axis.annotate('', (x * 1.25 - .02, y + .25), (x * 1.25 - .2, y + .25),
                          arrowprops={'arrowstyle': '->', 'lw': .9})


def search_chains():
    """How one candidate is trained and scored in Monte Carlo 1 and in Monte Carlo 2."""
    net, act, train, score, waste = '#eeeeee', '#cfe2f3', '#fce5cd', '#d9ead3', '#f4cccc'
    nl = chr(10)
    fig, axis = plt.subplots(figsize=(12, 7.2), layout='constrained')
    _chain(axis, 5.0, [('current' + nl + 'network', net), ('apply EVERY' + nl + 'candidate', waste),
                       ('train each copy' + nl + '1 epoch, low LR', waste),
                       ('pick one' + nl + 'candidate A', act), ('add a RANDOM' + nl + 'action', waste),
                       ('train' + nl + '1 epoch', train), ('add a RANDOM' + nl + 'action', waste),
                       ('train' + nl + '1 epoch', train), ('score' + nl + 'once', score)],
           'Monte Carlo 1: one score per visit, measured after random extra changes')
    axis.text(0, 4.55, 'All candidates are built and trained up front, even those never visited. The score of A is taken '
              'after two random later changes, so it mixes A with luck; values are then summed up the tree.',
              fontsize=8.5)
    _chain(axis, 2.6, [('current' + nl + 'network', net), ('apply' + nl + 'candidate A', act),
                       ('train, restarted' + nl + 'LR, seed 1', train), ('score A' + nl + '(last 2 epochs)', score),
                       ('train, restarted' + nl + 'LR, seed 2', train), ('score A' + nl + 'again', score),
                       ('A = average' + nl + 'of its scores', score)],
           'Monte Carlo 2: the candidate itself is scored, at least twice')
    _chain(axis, .9, [('trained copy' + nl + 'of A', train), ('apply next' + nl + 'action B', act),
                      ('train' + nl + '(continues A)', train), ('score' + nl + 'A then B', score)],
           'Monte Carlo 2, going deeper: the next step continues from the trained copy of A')
    axis.text(0, .35, 'Each candidate network is built only when it is first visited, every candidate in a search uses '
              'the same training seeds, and the winner is the candidate with the highest average.', fontsize=8.5)
    from matplotlib.patches import Patch
    axis.legend(handles=[Patch(facecolor=c, edgecolor='black', label=l) for c, l in
                         ((act, 'change to the network'), (train, 'training'), (score, 'score'),
                          (waste, 'wasted or random work'))], loc='upper right', fontsize=8, ncol=4,
                bbox_to_anchor=(1, 1.02))
    axis.set_xlim(-.1, 11.4)
    axis.set_ylim(0, 6.1)
    axis.axis('off')
    fig.savefig(OUTPUT / '013-search-chains.png', dpi=160)
    plt.close(fig)


def phase18_search():
    """Per search of the two surviving Phase 18 runs: candidates scored once or more often, and search time."""
    fig, (cover, time_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout='constrained')
    width = .38
    for offset, (version, seed, color) in enumerate(((1, 2, 'C0'), (2, 1, 'C1'))):
        searches = []
        for path in (RUN / f'mc_comparison_mc{version}/seed_{seed}/board/simulations').glob('simulation_gen_*.json'):
            search = json.loads(path.read_text(encoding='utf-8'))
            if search['generation'] < PHASE18_GENERATIONS - 1:
                searches.append(search)
        searches.sort(key=lambda search: search['generation'])
        x = [search['generation'] + (offset - .5) * width for search in searches]
        once = [sum(c['visits'] == 1 for c in search['candidates']) for search in searches]
        more = [sum(c['visits'] > 1 for c in search['candidates']) for search in searches]
        never = [sum(c['visits'] == 0 for c in search['candidates']) for search in searches]
        label = f'Monte Carlo {version} (seed {seed})'
        cover.bar(x, once, width, color=color, alpha=.55, label=f'{label}: scored once')
        cover.bar(x, more, width, bottom=once, color=color, label=f'{label}: scored 2 or more times')
        cover.bar(x, never, width, bottom=[a + b for a, b in zip(once, more)], color='red',
                  label='never scored' if offset == 0 else None)
        time_axis.plot([s['generation'] for s in searches], [s['durationSec'] / 60 for s in searches],
                       marker='o', color=color, label=label)
    time_axis.axhline(25, color='gray', ls='--', lw=1)
    time_axis.text(0, 27, 'nominal 25-minute cap', fontsize=8, color='gray')
    cover.set_ylabel('Candidates per search')
    cover.set_ylim(0, 55)
    cover.set_title('Phase 18: every candidate was scored, but later searches only once each', fontsize=11)
    cover.legend(fontsize=7, ncol=2)
    time_axis.set_ylabel('Search time (minutes)')
    time_axis.set_xlabel('Search after generation')
    time_axis.xaxis.set_major_locator(MultipleLocator(1))
    time_axis.legend(fontsize=8)
    for axis in (cover, time_axis):
        axis.grid(alpha=.25, axis='y')
    fig.savefig(OUTPUT / '013-phase18-search.png', dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    a08_leads()
    growth()
    phase13()
    phase15()
    phase16()
    search_chains()
    phase17()
    phase18()
    phase18_search()
    final_summary()


if __name__ == '__main__':
    main()
