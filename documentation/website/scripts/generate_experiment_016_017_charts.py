"""Generate training-history and parameter-count charts for the completed runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / 'experiments/output/train_cifar10/runs'
OUTPUT = ROOT / 'documentation/website/app/public/assets/experiments'
EPOCHS_PER_GENERATION = 20
GENERATIONS = 10
CASES = {
    '016': ('Budgeted search', RUNS / 'exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1'),
    '017': ('Family-balanced search', RUNS / 'exp012_family_search/v1/gen_10/family/cifar10/seed_1'),
}
# Action applied after each generation, from each run's board (listed in the 016 and 017 reports).
ACTIONS = {
    '016': ['+ Conv, BN, ReLU', '+ Residual conv', '+ Linear layer', '+ Dropout', 'Linear wider',
            '+ Dropout', '+ Linear layer', '+ Linear layer', '+ Residual conv'],
    '017': ['Widen 8→16', 'Widen 16→32', 'Widen 32→64', '+ Conv, BN, ReLU', '+ Linear layer',
            'Linear wider', 'Linear narrower', 'Linear wider', '+ Residual conv'],
}


def load_run(path):
    metrics = json.loads((path / 'board/metrics/training.json').read_text(encoding='utf-8'))['epochs']
    metadata = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    if not metrics:
        raise ValueError(f'No training history in {path}')
    return metrics, metadata


def setup_axis(axis):
    axis.grid(alpha=.25)


def mark_generations(axes, per_generation=EPOCHS_PER_GENERATION, generations=GENERATIONS):
    total = per_generation * generations
    for axis in axes:
        axis.set_xlim(0.5, total + 0.5)
        axis.xaxis.set_major_locator(MultipleLocator(per_generation))
        axis.grid(False, axis='x')
        for boundary in range(per_generation, total, per_generation):
            axis.axvline(boundary + 0.5, color='gray', alpha=.5, lw=0.8, zorder=0)
    top = axes[0].secondary_xaxis('top')
    top.set_xticks([per_generation * (g + .5) for g in range(generations)], [str(g) for g in range(generations)])
    top.tick_params(length=0, labelsize=8)
    top.set_xlabel('Generation', fontsize=9)


def label_actions(axis, actions, per_generation=EPOCHS_PER_GENERATION):
    for generation, action in enumerate(actions):
        axis.text(per_generation * (generation + 1) + 1, 4, action, rotation=90, fontsize=8,
                  va='bottom', ha='left')


def individual(identifier, name, metrics, metadata):
    x = [row['globalEpoch'] + 1 for row in metrics]
    fig, (acc, size) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    fig.suptitle(f'Experiment {identifier}: {name} (CIFAR-10, seed 1)', fontsize=14, weight='bold')
    acc.plot(x, [100 * row['trainAcc'] for row in metrics], label='Training')
    acc.plot(x, [100 * row['valAcc'] for row in metrics], label='Validation')
    acc.set_ylabel('Accuracy (%)')
    acc.set_ylim(0, 100)
    acc.legend(loc='upper left', frameon=False, ncol=2)
    acc.set_title(f"Final validation: {metadata['final_validation_accuracy'] * 100:.2f}%  |  "
                  f"Final test: {metadata['final_test_accuracy'] * 100:.2f}%", fontsize=10)
    size.step(x, [row['paramCount'] for row in metrics], where='post', color='C2')
    size.set_xlabel('Training epoch')
    size.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    size.set_yscale('log')
    size.set_ylabel('Parameters (log scale)')
    for axis in (acc, size):
        setup_axis(axis)
    mark_generations((acc, size))
    label_actions(acc, ACTIONS.get(identifier, []))
    fig.savefig(OUTPUT / f'{identifier}-training-and-parameters.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def comparison(runs):
    fig, (acc, size) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    fig.suptitle('CIFAR-10: validation accuracy and model growth', fontsize=14, weight='bold')
    for identifier, color in [('016', 'C0'), ('017', 'C1')]:
        name, metrics, _ = runs[identifier]
        x = [row['globalEpoch'] + 1 for row in metrics]
        label = f'{identifier}: {name}'
        acc.plot(x, [100 * row['valAcc'] for row in metrics], label=label, color=color)
        size.step(x, [row['paramCount'] for row in metrics], where='post', color=color, label=label)
    acc.set_ylim(0, 100)
    acc.set_ylabel('Validation accuracy (%)')
    acc.legend(loc='lower right', frameon=False)
    size.set_yscale('log')
    size.set_ylabel('Parameters (log scale)')
    size.set_xlabel('Training epoch')
    size.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    for axis in (acc, size):
        setup_axis(axis)
    mark_generations((acc, size))
    fig.savefig(OUTPUT / '016-017-validation-and-growth.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for identifier, (name, path) in CASES.items():
        metrics, metadata = load_run(path)
        runs[identifier] = (name, metrics, metadata)
        individual(identifier, name, metrics, metadata)
    comparison(runs)
    print(f'Wrote three charts to {OUTPUT}')


if __name__ == '__main__':
    main()
