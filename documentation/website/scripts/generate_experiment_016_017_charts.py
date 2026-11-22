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
CASES = {
    '012': ('Budgeted search', RUNS / 'exp011_budgeted_mcts/v3/gen_10/budgeted_wide_direct/cifar10/seed_1'),
    '013': ('Family-balanced search', RUNS / 'exp012_family_search/v1/gen_10/family/cifar10/seed_1'),
}


def load_run(path):
    metrics = json.loads((path / 'board/metrics/training.json').read_text(encoding='utf-8'))['epochs']
    metadata = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    if not metrics:
        raise ValueError(f'No training history in {path}')
    return metrics, metadata


def setup_axis(axis):
    axis.grid(True, color='#d5dbe5', alpha=0.7, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def generation_lines(axes, epochs, per_generation=20):
    for boundary in range(per_generation, epochs, per_generation):
        for axis in axes:
            axis.axvline(boundary + 0.5, color='#9ba7b8', linestyle=':', linewidth=0.8)


def individual(identifier, name, metrics, metadata):
    x = [row['globalEpoch'] + 1 for row in metrics]
    fig, (acc, size) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    fig.suptitle(f'Experiment {identifier}: {name} — CIFAR-10, seed 1', fontsize=14, weight='bold')
    acc.plot(x, [100 * row['trainAcc'] for row in metrics], color='#2563eb', lw=1.6, label='Training')
    acc.plot(x, [100 * row['valAcc'] for row in metrics], color='#e56b28', lw=1.6, label='Validation')
    acc.set_ylabel('Accuracy (%)')
    acc.set_ylim(0, 100)
    acc.legend(loc='lower right', frameon=False, ncol=2)
    acc.set_title(f"Final validation: {metadata['final_validation_accuracy'] * 100:.2f}%  |  "
                  f"Final test: {metadata['final_test_accuracy'] * 100:.2f}%", fontsize=10)
    size.step(x, [row['paramCount'] for row in metrics], where='post', color='#168a69', lw=1.8)
    size.set_ylabel('Parameters')
    size.set_xlabel('Training epoch')
    size.xaxis.set_major_locator(MultipleLocator(20))
    size.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    for axis in (acc, size):
        setup_axis(axis)
        axis.set_xlim(1, x[-1])
    generation_lines((acc, size), x[-1])
    fig.text(0.01, -0.01, 'Dotted lines mark generation boundaries; parameter count is recorded during training.',
             fontsize=8, color='#536176')
    fig.savefig(OUTPUT / f'{identifier}-training-and-parameters.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def comparison(runs):
    fig, (acc, size) = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    fig.suptitle('CIFAR-10: validation accuracy and model growth', fontsize=14, weight='bold')
    for identifier, color in [('012', '#2563eb'), ('013', '#e56b28')]:
        name, metrics, _ = runs[identifier]
        x = [row['globalEpoch'] + 1 for row in metrics]
        label = f'{identifier}: {name}'
        acc.plot(x, [100 * row['valAcc'] for row in metrics], label=label, color=color, lw=1.7)
        size.step(x, [row['paramCount'] for row in metrics], where='post', color=color, lw=1.7, label=label)
    acc.set_ylim(0, 100)
    acc.set_ylabel('Validation accuracy (%)')
    acc.legend(loc='lower right', frameon=False)
    size.set_yscale('log')
    size.set_ylabel('Parameters (log scale)')
    size.set_xlabel('Training epoch')
    size.xaxis.set_major_locator(MultipleLocator(20))
    size.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    for axis in (acc, size):
        setup_axis(axis)
        axis.set_xlim(1, max(row['globalEpoch'] + 1 for row in runs['013'][1]))
    generation_lines((acc, size), len(runs['013'][1]))
    fig.text(0.01, -0.01, 'One seed per method; implementations differ. Dotted lines mark generation boundaries.',
             fontsize=8, color='#536176')
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
