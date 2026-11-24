"""Visualize completed Experiment 013 seed 1 using its saved board records."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key, candidate_detail, GROUPS, COLORS

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1/full_growth_5min/seed_1'
OUTPUT = ROOT / 'documentation/website/app/public/assets/experiments'


def generate():
    if not (RUN / 'run.json').exists():
        return []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RUN / 'board/metrics/training.json').read_text(encoding='utf-8'))['epochs']
    searches = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(
        (RUN / 'board/simulations').glob('simulation_gen_*.json'),
        key=lambda p: int(p.stem.rsplit('_', 1)[1]))]
    epochs = [r['globalEpoch'] + 1 for r in metrics]
    fig, (ax, size) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    ax.plot(epochs, [100*r['trainAcc'] for r in metrics], label='Training', color='#2563eb')
    ax.plot(epochs, [100*r['valAcc'] for r in metrics], label='Validation', color='#e56b28')
    ax.set(ylabel='Accuracy (%)', ylim=(0, 100), title='Experiment 013 Phase 5 · seed 1')
    ax.legend()
    size.step(epochs, [r['paramCount'] / 1000 for r in metrics], where='post', color='#168a69')
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    size.xaxis.set_major_locator(MultipleLocator(20))
    size.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    for axis in (ax, size):
        axis.grid(alpha=.25)
        for boundary in range(20, 200, 20):
            axis.axvline(boundary + .5, color='#999999', ls=':', lw=.8)
    fig.savefig(OUTPUT / '018-full-seed1-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    groups = [(key, name.replace('\n', ' ')) for key, name in GROUPS
              if any(group_key(row) == key for s in searches for row in s['candidates'])]
    ids = {}
    for s in searches:
        for row in s['candidates']:
            ids.setdefault(row['action'], f'A{len(ids)+1:03d}')
    with (OUTPUT / '018-full-seed1-candidates.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'action_id', 'action_type', 'full_recorded_action',
                         'mean_search_score', 'visits', 'chosen', 'last_rollout_validation_accuracy'])
        for s in searches:
            for r in s['candidates']:
                writer.writerow([s['generation'], ids[r['action']], group_key(r), r['action'],
                                 r.get('score'), r.get('visits'), r.get('chosen'), r.get('accuracyAfter')])

    fig, axes = plt.subplots(len(searches), 1, figsize=(15, 2.8 * len(searches)), squeeze=False,
                              sharex=True, sharey=True)
    fig.suptitle('Phase 5 seed 1: individual candidate search scores by family', fontsize=15, weight='bold')
    for ax, s in zip(axes.flat, searches):
        for position, (key, _) in enumerate(groups):
            rows = [r for r in s['candidates'] if group_key(r) == key and r.get('score') is not None]
            for j, row in enumerate(rows):
                offset = .56 * (j - (len(rows)-1)/2) / max(len(rows)-1, 1)
                color = COLORS['Sequential convolution'] if key.startswith('Sequential convolution') else COLORS.get(key, COLORS['Other'])
                ax.scatter(position + offset, row['score'], s=110 if row.get('chosen') else 24,
                           marker='*' if row.get('chosen') else 'o', color=color,
                           edgecolors='black' if row.get('chosen') else 'none', zorder=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Gen {s['generation']}\nscore")
        ax.grid(axis='y', alpha=.25)
    axes[-1, 0].set_xticks(range(len(groups)), [label for _, label in groups], rotation=25, ha='right')
    fig.subplots_adjust(top=.97, bottom=.035, hspace=.22)
    fig.savefig(OUTPUT / '018-full-seed1-scores-overview.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    for s in searches:
        order = {key: i for i, (key, _) in enumerate(groups)}
        rows = sorted((r for r in s['candidates'] if r.get('score') is not None),
                      key=lambda r: (order[group_key(r)], candidate_detail(r), r['action']))
        fig, ax = plt.subplots(figsize=(14, max(4, len(rows)*.28 + 2)), layout='constrained')
        colors = [COLORS['Sequential convolution'] if group_key(r).startswith('Sequential convolution')
                  else COLORS.get(group_key(r), COLORS['Other']) for r in rows]
        positions = list(range(len(rows)))
        ax.barh(positions, [r['score'] for r in rows], color=colors,
                edgecolor=['black' if r.get('chosen') else 'none' for r in rows],
                linewidth=[1.5 if r.get('chosen') else 0 for r in rows])
        labels = [f"{ids[r['action']]} · {group_key(r)} · {candidate_detail(r).replace(chr(10), ' / ')}" for r in rows]
        ax.set_yticks(positions, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set(xlim=(0, 1.06), xlabel='Mean search score',
               title=f"Generation {s['generation']} · {len(rows)} evaluated candidates · ★ selected")
        ax.grid(axis='x', alpha=.25)
        ax.set_axisbelow(True)
        for pos, row in enumerate(rows):
            ax.text(min(row['score'] + .008, .98), pos,
                    f"{'★ ' if row.get('chosen') else ''}{row['score']:.3f}  n={row.get('visits', 0)}",
                    va='center', fontsize=7)
        fig.savefig(OUTPUT / f"018-full-seed1-scores-gen-{s['generation']}.png", dpi=140, bbox_inches='tight')
        plt.close(fig)

    lines = ['', '### Seed 1: training history and action decisions', '',
             '![Seed 1 training accuracy and model parameters](/assets/experiments/018-full-seed1-training.png)', '',
             'The dotted lines mark generation boundaries. The change after 20 epochs includes the '
             'selected action **and continued training**; it is not an isolated causal effect.', '',
             '| Generation ending | Before action validation | Parameters | Selected action | Next generation validation | Change after training | Candidates | Search time |',
             '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    for s in searches:
        gen = s['generation']
        before = [r for r in metrics if r['generation'] == gen][-1]
        later = [r for r in metrics if r['generation'] == gen + 1][-1]
        chosen = next(r for r in s['candidates'] if r.get('chosen'))
        label = f"{ids[chosen['action']]}: {group_key(chosen)} — {candidate_detail(chosen).replace(chr(10), ' / ')}"
        lines.append(f"| {gen} (epoch {(gen+1)*20}) | {before['valAcc']*100:.2f}% | {before['paramCount']:,} | "
                     f"{label} | {later['valAcc']*100:.2f}% | {(later['valAcc']-before['valAcc'])*100:+.2f} pp | "
                     f"{len(s['candidates'])} | {s['durationSec']/60:.1f} min |")
    lines += ['', '### Seed 1: distribution of candidate scores', '',
              'Each dot is one recorded candidate; the black-edged star is the chosen action. Family '
              'positions stay fixed in every generation. These are search scores, not the individual '
              'accuracy measurements inside each rollout. One visit can combine multiple model scores; '
              'the CSV includes the board’s separate latest rollout accuracy.', '',
              '![Seed 1 candidate score distributions](/assets/experiments/018-full-seed1-scores-overview.png)', '',
              'Individual action charts: ' + ', '.join(
                  f"[generation {s['generation']}](/assets/experiments/018-full-seed1-scores-gen-{s['generation']}.png)"
                  for s in searches) + '.', '',
              '[Download all recorded candidate scores and full action descriptions](/assets/experiments/018-full-seed1-candidates.csv).', '',
              '| Generation | ' + ' | '.join(label for _, label in groups) + ' | Total scored |',
              '| ---: | ' + ' | '.join('---:' for _ in groups) + ' | ---: |']
    for s in searches:
        counts = Counter(group_key(r) for r in s['candidates'] if r.get('score') is not None)
        lines.append(f"| {s['generation']} | " + ' | '.join(str(counts[key]) for key, _ in groups) +
                     f" | {sum(counts.values())} |")
    lines += ['', 'Counts are scored candidates, not proof of how many were generated or available. '
              'Blank or zero means none of that type is recorded as scored.', '']
    return lines
