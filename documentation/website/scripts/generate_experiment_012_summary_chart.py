"""Summary chart for Experiment 012: where the residual-block action ended up.

Left: validation gain over the unchanged network after one generation, per duplicate version (Part 4 A,
plain checkpoint), from data/experiments/experiment-015-duplicate-action-design.json.
Right: mean final CIFAR-10 test accuracy of the Part 2 growing runs (three seeds), per initializer and
without block actions, as published in Part 2 of the report.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SITE = Path(__file__).resolve().parents[1]
DATA = SITE / 'data/experiments/experiment-015-duplicate-action-design.json'
OUTPUT = SITE / 'app/public/assets/experiments/012-summary.png'
VERSIONS = ['current', '+side', '+after_relu']
PART2_CIFAR10 = [('Copy', 61.20), ('Quasi-identity', 62.24), ('Residual-zero', 54.66), ('Block actions off', 57.40)]


def main():
    data = json.loads(DATA.read_text(encoding='utf-8'))
    rows = {row['setting']: row for row in data['paired']['tables']['Grow ladder, plain checkpoint']}
    base = rows['no change']['val_after_generation']
    gains = [100 * (rows[v]['val_after_generation'] - base) for v in VERSIONS]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    bars = left.bar(VERSIONS, gains, color=['C7', 'C7', 'C2'])
    left.bar_label(bars, [f'+{g:.1f} pp' for g in gains])
    left.axhline(0, color='black', lw=.8)
    left.set_ylabel('Validation gain over no change (pp)')
    left.set_title('Applied by hand: the final version helps most', fontsize=11)
    left.set_ylim(0, max(gains) * 1.25)
    left.grid(alpha=.25, axis='y')

    names = [name for name, _ in PART2_CIFAR10]
    values = [value for _, value in PART2_CIFAR10]
    bars = right.bar(names, values, color=['C7', 'C2', 'C7', 'C0'])
    right.bar_label(bars, [f'{v:.1f}%' for v in values])
    right.set_ylabel('Mean final test accuracy (%)')
    right.set_title('Growing runs on CIFAR-10: quasi-identity ends best', fontsize=11)
    right.set_ylim(50, 66)
    right.grid(alpha=.25, axis='y')
    fig.savefig(OUTPUT, dpi=160)
    plt.close(fig)
    print(dict(zip(VERSIONS, gains)))


if __name__ == '__main__':
    main()
