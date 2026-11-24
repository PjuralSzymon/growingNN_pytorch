"""Phase 18, Monte Carlo 2 seed 1: how the network grew, and what each selected action did.

1. Structure snapshots: the board's own graph drawings (draw_filtered_fx_graph, saved by the run after
   every generation) for the start, two generations in between and the last generation shown in the report.
2. Action summary: for each search, the selected action, the parameters it added and the change in
   training and validation accuracy from the end of that generation to the end of the next one.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key  # noqa: E402

BOARD = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1/mc_comparison_mc2/seed_1/board'
OUTPUT = ROOT / 'documentation/website/app/public/assets/experiments'
EPOCHS_PER_GENERATION = 30
REPORT_EPOCHS = 300  # the report shows the first 300 epochs (10 generations)
SNAPSHOTS = (0, 3, 6, 9)


def generation_ends(epochs):
    """Training/validation accuracy and parameters at the last epoch of every finished generation."""
    ends = {}
    for row in epochs:
        if row['globalEpoch'] + 1 <= REPORT_EPOCHS and row['epochInGeneration'] == EPOCHS_PER_GENERATION - 1:
            ends[row['generation']] = row
    return ends


def structures(epochs):
    params = {}
    for row in epochs:
        params.setdefault(row['generation'], row['paramCount'])
    for generation in SNAPSHOTS:
        page = pymupdf.open(BOARD / f'graphs/gen_{generation}_simplified.pdf')[0]
        page.get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(OUTPUT / f'013-mc2-structure-gen{generation}.png')
        print(f'generation {generation}: {params.get(generation):,} parameters')
    return params


def actions(ends):
    rows = []
    for generation in sorted(ends):
        after = ends.get(generation + 1)
        search = BOARD / f'simulations/simulation_gen_{generation}.json'
        if after is None or not search.is_file():
            continue
        chosen = json.loads(search.read_text(encoding='utf-8'))['actionChosen']
        before = ends[generation]
        rows.append((generation, group_key({'action': chosen}).replace('Sequential convolution', 'Seq. conv'), after['paramCount'] - before['paramCount'],
                     100 * (after['trainAcc'] - before['trainAcc']), 100 * (after['valAcc'] - before['valAcc'])))
    labels = [f'after gen {g}\n{name}\n+{added / 1000:.0f}k params' for g, name, added, _, _ in rows]
    x = range(len(rows))
    fig, axis = plt.subplots(figsize=(12, 4.5), layout='constrained')
    width = .38
    train = axis.bar([i - width / 2 for i in x], [r[3] for r in rows], width, label='Training accuracy change')
    val = axis.bar([i + width / 2 for i in x], [r[4] for r in rows], width, label='Validation accuracy change')
    axis.bar_label(train, fmt='%+.1f', fontsize=7)
    axis.bar_label(val, fmt='%+.1f', fontsize=7)
    axis.axhline(0, color='black', lw=.8)
    axis.set_xticks(list(x), labels, fontsize=7)
    axis.set_ylabel('Change over the next generation (pp)')
    axis.set_title('Phase 18, Monte Carlo 2 seed 1: each selected action and what followed it', fontsize=11)
    axis.grid(alpha=.25, axis='y')
    axis.legend(fontsize=8)
    fig.savefig(OUTPUT / '013-mc2-actions.png', dpi=160)
    plt.close(fig)
    for row in rows:
        print(row)
    action_types(rows)


def action_types(rows):
    """Experiment 002 Figure 4 style: mean change per action type (bars) and every single action (dots)."""
    types = sorted({row[1] for row in rows})
    fig, axis = plt.subplots(figsize=(10, 1.4 + 1.1 * len(types)), layout='constrained')
    height = .38
    for i, name in enumerate(types):
        train = [row[3] for row in rows if row[1] == name]
        val = [row[4] for row in rows if row[1] == name]
        axis.barh(i - height / 2, sum(train) / len(train), height, color='#a9bfdc',
                  label='Training accuracy' if i == 0 else None)
        axis.barh(i + height / 2, sum(val) / len(val), height, color='#b7d3be',
                  label='Validation accuracy' if i == 0 else None)
        axis.scatter(train, [i - height / 2] * len(train), s=18, color='#4c6a9c', zorder=3)
        axis.scatter(val, [i + height / 2] * len(val), s=18, color='#5f8a6a', zorder=3)
    axis.set_yticks(range(len(types)), [f'{name} ({sum(row[1] == name for row in rows)}x)' for name in types])
    axis.axvline(0, color='black', lw=.8)
    axis.set_xlabel('Accuracy change over the next generation (percentage points)')
    axis.set_title('Monte Carlo 2 seed 1: training- and validation-accuracy change by action type', fontsize=11)
    axis.grid(alpha=.3, axis='x')
    axis.legend(fontsize=8, loc='lower right')
    fig.savefig(OUTPUT / '013-mc2-action-types.png', dpi=160)
    plt.close(fig)


def main():
    epochs = json.loads((BOARD / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']
    structures(epochs)
    actions(generation_ends(epochs))


if __name__ == '__main__':
    main()
