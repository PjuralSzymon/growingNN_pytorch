"""Plot the recorded root-candidate mean scores for Experiment 017 seed 1.

Each bar is an evaluated candidate, not an individual rollout. Candidates
absent from the board were not necessarily illegal; they may not have been
evaluated before the search stopped. No simulation occurred after generation 9.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BOARD = ROOT / 'experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1/board'
DEFAULT_OUTPUT = ROOT / 'documentation/website/app/public/assets/experiments'
COLORS = {
    'Widen convolution': '#3576c8',
    'Sequential convolution': '#db6b27',
    'Residual convolution': '#a850a2',
    'Sequential linear': '#168a69',
    'Add normalization': '#c2357a',
    'Add neurons': '#628f28',
    'Delete neurons': '#b48735',
    'Dropout': '#718096',
    'Delete layer': '#aa5961',
    'Other': '#888888',
}


def classify(row):
    text = row.get('action') or ''
    name = row.get('name') or ''
    if 'Widen Conv' in text:
        return 'Widen convolution'
    if 'Add Normalization' in text:
        return 'Add normalization'
    if 'Add Seq Conv' in text:
        return 'Sequential convolution'
    if 'Add Res Conv' in text:
        return 'Residual convolution'
    if 'Add Seq Linear' in text:
        return 'Sequential linear'
    if 'Add Neurons' in text:
        return 'Add neurons'
    if 'Delete Neurons' in text or 'Del Neurons' in text:
        return 'Delete neurons'
    if 'Dropout' in text:
        return 'Dropout'
    if 'Delete Layer' in text or 'Del Layer' in text:
        return 'Delete layer'
    return name or 'Other'


def label(row):
    text = row.get('action') or ''
    family = classify(row)
    pair = re.search(r"\['([^']+)', '([^']+)'", text)
    location = f'{pair.group(1)} → {pair.group(2)}' if pair else ''
    if family == 'Widen convolution':
        match = re.search(r"\['([^']+)',\s*(\d+)", text)
        detail = f'{match.group(1)} → {match.group(2)} ch' if match else text
        return f'Widen conv\n{detail}'
    if family in ('Add neurons', 'Delete neurons'):
        match = re.search(r"\['([^']+)',\s*([0-9.]+)", text)
        detail = f'{match.group(1)} ×{match.group(2)}' if match else text
        return f'{family}\n{detail}'
    if family == 'Sequential convolution':
        has_bn = bool(re.search(r',\s*(True|False),\s*True,\s*(True|False)\]\s*\)', text))
        return f'Seq conv\n{location}\n' + ('BN + ReLU' if has_bn else 'no BN/ReLU')
    if family == 'Dropout':
        match = re.search(r'Dropout\w*\(p=([0-9.]+)', text)
        return f'Dropout {match.group(1) if match else "?"}\n{location}'
    short = {'Residual convolution': 'Residual conv', 'Sequential linear': 'Seq linear',
             'Delete layer': 'Delete layer'}.get(family, family)
    if family == 'Delete layer' and not location:
        match = re.search(r"\['([^']+)'", text)
        location = match.group(1) if match else ''
    return f'{short}\n{location}'.strip()


def read_searches(board):
    paths = sorted((board / 'simulations').glob('simulation_gen_*.json'),
                   key=lambda p: int(p.stem.rsplit('_', 1)[1]))
    result = []
    for path in paths:
        record = json.loads(path.read_text(encoding='utf-8'))
        rows = [row for row in record.get('candidates', []) if row.get('score') is not None]
        rows.sort(key=lambda row: (list(COLORS).index(classify(row)) if classify(row) in COLORS else len(COLORS),
                                   label(row), row.get('action') or ''))
        if rows:
            result.append((record, rows))
    return result


GROUPS = [
    ('Widen convolution', 'Widen\nconvolution'),
    ('Sequential convolution + BN', 'Sequential conv\nwith BN + ReLU'),
    ('Sequential convolution without BN', 'Sequential conv\nwithout BN/ReLU'),
    ('Residual convolution', 'Residual\nconvolution'),
    ('Sequential linear', 'Sequential\nlinear'),
    ('Add normalization', 'Add\nnormalization'),
    ('Add neurons', 'Add\nneurons'),
    ('Delete neurons', 'Delete\nneurons'),
    ('Dropout', 'Dropout'),
    ('Delete layer', 'Delete\nlayer'),
    ('Other', 'Other'),
]


def group_key(row):
    family = classify(row)
    if family == 'Sequential convolution':
        return ('Sequential convolution without BN' if 'no BN/ReLU' in label(row)
                else 'Sequential convolution + BN')
    return family if family in dict(GROUPS) else 'Other'


def action_identity(row):
    """Keep concrete action parameters; omit widening's per-evaluation RNG seed."""
    text = ' '.join((row.get('action') or '').split())
    if group_key(row) == 'Widen convolution':
        text = re.sub(r"(Widen Conv Action: \['[^']+',\s*\d+),\s*\d+(\])", r'\1\2', text)
    return text


def group_layout(searches):
    """Give each distinct recorded action its own fixed x position."""
    actions = {key: {} for key, _ in GROUPS}
    for _, rows in searches:
        for row in rows:
            actions[group_key(row)].setdefault(action_identity(row), row)
    layout = {}
    offset = 0.0
    code = 1
    for key, title in GROUPS:
        if not actions[key]:
            continue
        if key == 'Widen convolution':
            def sort_key(identity):
                match = re.search(r"\['([^']+)',\s*(\d+)", identity)
                return (match.group(1), int(match.group(2))) if match else ('', 0)
            identities = sorted(actions[key], key=sort_key)
        else:
            identities = sorted(actions[key], key=lambda identity: (candidate_detail(actions[key][identity]), identity))
        positions = [offset + index for index in range(len(identities))]
        codes = [f'A{index:02d}' for index in range(code, code + len(identities))]
        layout[key] = {'title': title, 'positions': positions, 'identities': identities,
                       'examples': [actions[key][identity] for identity in identities], 'codes': codes}
        code += len(identities)
        offset = positions[-1] + 1.8
    return layout


def placed_rows(rows, layout):
    grouped = {key: {} for key in layout}
    for row in rows:
        key, identity = group_key(row), action_identity(row)
        if identity in grouped[key]:
            raise ValueError(f'Duplicate action identity in one generation: {identity}')
        grouped[key][identity] = row
    placed = []
    for key, group in layout.items():
        for position, identity in zip(group['positions'], group['identities']):
            if identity in grouped[key]:
                placed.append((position, grouped[key][identity]))
    return placed


def candidate_detail(row):
    lines = label(row).splitlines()
    if group_key(row).startswith('Sequential convolution'):
        match = re.search(r'Conv2d\((\d+),\s*(\d+),\s*kernel_size=\((\d+),\s*(\d+)\)', row.get('action') or '')
        size = f'{match.group(1)}→{match.group(2)}, {match.group(3)}×{match.group(4)}' if match else ''
        return (lines[1] if len(lines) > 1 else '') + '\n' + size
    if group_key(row) in ('Residual convolution', 'Sequential linear'):
        text = row.get('action') or ''
        conv = re.search(r'Conv2d\((\d+),\s*(\d+),\s*kernel_size=\((\d+),\s*(\d+)\)', text)
        linear = re.search(r'Linear\(in_features=(\d+),\s*out_features=(\d+)', text)
        dims = (f'{conv.group(1)}→{conv.group(2)}, {conv.group(3)}×{conv.group(4)}' if conv else
                f'{linear.group(1)}→{linear.group(2)}' if linear else '')
        return (lines[1] if len(lines) > 1 else '') + '\n' + dims
    if group_key(row) == 'Dropout':
        return '\n'.join(lines)
    return '\n'.join(lines[1:]) if len(lines) > 1 else lines[0]


def draw(ax, record, rows, layout):
    placed = placed_rows(rows, layout)
    positions = [position for position, _ in placed]
    selected_rows = [row for _, row in placed]
    bars = ax.bar(positions, [row['score'] for row in selected_rows], width=.72,
                  color=[COLORS.get(classify(row), COLORS['Other']) for row in selected_rows],
                  edgecolor=['#111827' if row.get('chosen') else 'none' for row in selected_rows],
                  linewidth=[2.0 if row.get('chosen') else 0 for row in selected_rows])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Mean score (0–1)', fontsize=10)
    ax.set_title(f"Generation {record['generation']}  ·  after epoch {(record['generation'] + 1) * 20}",
                 loc='left', fontsize=13, weight='bold', y=1.17)
    ticks = [position for group in layout.values() for position in group['positions']]
    codes = {position: code for group in layout.values()
             for position, code in zip(group['positions'], group['codes'])}
    names = {position: codes[position] + '\n' + candidate_detail(row) for position, row in placed}
    ax.set_xticks(ticks)
    ax.set_xticklabels([names.get(position, '') for position in ticks], rotation=45, ha='right', fontsize=7.5)
    ax.set_xlim(ticks[0] - .7, ticks[-1] + .7)
    for key, group in layout.items():
        center = (group['positions'][0] + group['positions'][-1]) / 2
        color = COLORS['Sequential convolution'] if key.startswith('Sequential convolution') else COLORS.get(key, COLORS['Other'])
        ax.text(center, 1.02, group['title'], transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8.5, color=color, weight='bold')
    ax.grid(axis='y', alpha=.28)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='y', labelsize=9)
    for bar, row in zip(bars, selected_rows):
        score = row['score']
        visits = row.get('visits', 0)
        marker = '★ ' if row.get('chosen') else ''
        ax.text(bar.get_x() + bar.get_width()/2, min(score + .02, .98),
                f'{marker}{score:.3f}\nn={visits}', ha='center', va='bottom',
                fontsize=9, color='#111827')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', type=Path, default=DEFAULT_BOARD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    searches = read_searches(args.board)
    if not searches:
        raise SystemExit(f'No scored candidates found in {args.board}')
    args.output.mkdir(parents=True, exist_ok=True)
    layout = group_layout(searches)
    with (args.output / '017-action-score-details.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['generation', 'action_id', 'action_type', 'full_recorded_action', 'mean_score', 'visits', 'chosen'])
        codes = {(key, identity): code for key, group in layout.items()
                 for identity, code in zip(group['identities'], group['codes'])}
        for record, rows in searches:
            for row in rows:
                writer.writerow([record['generation'], codes[(group_key(row), action_identity(row))],
                                 group_key(row), row.get('action'), row['score'], row.get('visits'), row.get('chosen')])
    for record, rows in searches:
        fig, ax = plt.subplots(figsize=(36, 7), layout='constrained')
        draw(ax, record, rows, layout)
        fig.suptitle('Experiment 017: evaluated root candidates · CIFAR-10 seed 1', fontsize=12)
        fig.text(.01, -.01, '★ / black outline: chosen action. n: visits. Only evaluated candidates are shown.',
                 fontsize=9, color='#4b5563')
        fig.savefig(args.output / f'017-action-scores-gen-{record["generation"]}.png', dpi=170, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(len(searches), 1, figsize=(36, 5.2 * len(searches)), squeeze=False)
    for ax, (record, rows) in zip(axes.flat, searches):
        draw(ax, record, rows, layout)
    fig.subplots_adjust(left=.035, right=.995, top=.95, bottom=.015, hspace=1.35)
    fig.suptitle('Experiment 017: evaluated candidate scores by generation · seed 1\n'
                 '★ / black outline = chosen action; n = visits; only evaluated candidates shown',
                 fontsize=15, weight='bold', y=.997)
    fig.savefig(args.output / '017-action-scores-overview.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote overview and {len(searches)} generation charts to {args.output}')


if __name__ == '__main__':
    main()
