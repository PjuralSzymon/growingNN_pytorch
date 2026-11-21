"""Create Experiment 014's report and learning-curve figure from saved results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1'
REPORT = ROOT / 'documentation/website/content/experiments/experiment-014-action-replay.md'
CHART = ROOT / 'documentation/website/app/public/assets/experiments/014-action-replay-validation.png'
SCORING_CHART = ROOT / 'documentation/website/app/public/assets/experiments/014-scoring-epochs.png'
MCTS_CHART = ROOT / 'documentation/website/app/public/assets/experiments/014-montecarlo2-epochs.png'
TIME_CHART = ROOT / 'documentation/website/app/public/assets/experiments/014-montecarlo2-time.png'
LABELS = {'control': 'Unchanged', 'candidate_1': 'Widen to 256', 'candidate_2': 'Widen to 128',
          'candidate_3': 'Add linear', 'candidate_4': 'Conv without BN/ReLU',
          'candidate_5': 'A08: conv + BN + ReLU'}


def scoring_section():
    path = RUN / 'scoring_epochs/summary.json'
    values = {}
    if path.exists():
        values = {(row['epochs'], row['arm']): row['score']
                  for row in json.loads(path.read_text(encoding='utf-8'))['results']}
        fig, ax = plt.subplots(figsize=(10, 5.5))
        epochs_list = (1, 5, 10, 20)
        for i in range(1, 6):
            arm = f'candidate_{i}'
            ax.plot(epochs_list, [100 * values[(epoch, arm)] for epoch in epochs_list],
                    marker='o', linewidth=2.8 if i == 5 else 1.5, label=LABELS[arm])
        ax.set(xlabel='Scoring epochs', ylabel='Simulation validation accuracy (%)',
               title='Experiment 14: effect of scoring epochs', xticks=epochs_list)
        ax.set_ylim(0, 100)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        SCORING_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(SCORING_CHART, dpi=150)
        plt.close(fig)
    lines = ['', '## Phase 2: scoring epochs', '',
             'Each cell is validation accuracy after scoring a fresh copy of the same action, '
             'using the original simulation subset and the Monte Carlo rollout scoring path with '
             'rollout depth zero. The epoch settings are separate runs, using the same nominal seed; '
             'one score per action is not a stability estimate.', '',
             '| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for epochs in (1, 5, 10, 20):
        cells = [f"{100 * values[(epochs, f'candidate_{i}')]:.2f}%" if (epochs, f'candidate_{i}') in values
                 else 'Pending' for i in range(1, 6)]
        lines.append(f"| {epochs} | " + ' | '.join(cells) + ' |')
    note = ('These simulation scores use different data and training budgets than the full-training results above.'
            if values else 'Run the experiment script with `--phase scoring`, then regenerate this report.')
    if values:
        lines += ['', '![Simulation accuracy by scoring epochs and action](/assets/experiments/014-scoring-epochs.png)']
    return lines + ['', note, '']


def montecarlo2_section():
    target = RUN / 'montecarlo2_epochs'
    plotted = {}
    lines = ['', '## Phase 3: Monte Carlo 2 search scores', '',
             'Each row runs the complete Monte Carlo 2 search from the same checkpoint with the five recorded '
             'root actions. Cells show the final root-node mean search value and visit count; **bold** '
             'marks the selected action. These values combine rollout scores and are not necessarily '
             'direct validation accuracies. The search can include randomly chosen future actions. '
             'The configured 32-second budget is soft and may be exceeded while completing search.', '',
             '| Scoring epochs | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |',
             '| ---: | ---: | ---: | ---: | ---: | ---: |']
    for epochs in (1, 5, 10, 20):
        path = target / f'epochs_{epochs}.json'
        if not path.exists():
            lines.append(f'| {epochs} | Pending | Pending | Pending | Pending | Pending |')
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        by_arm = {row['arm']: row for row in data['results']}
        plotted[epochs] = by_arm
        cells = []
        for i in range(1, 6):
            row = by_arm[f'candidate_{i}']
            cell = f"{row['score']:.4f} (n={row['visits']})" if row['score'] is not None else 'Not scored'
            cells.append(f'**{cell}**' if row['chosen'] else cell)
        lines.append(f"| {epochs} | " + ' | '.join(cells) + ' |')
    if plotted:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i in range(1, 6):
            arm = f'candidate_{i}'
            epochs = sorted(plotted)
            ax.plot(epochs, [plotted[e][arm]['score'] for e in epochs], marker='o',
                    linewidth=2.8 if i == 5 else 1.5, label=LABELS[arm])
        ax.set(xlabel='Scoring epochs', ylabel='Mean search value',
               title='Experiment 14: Monte Carlo 2 search values', xticks=sorted(plotted))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        MCTS_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(MCTS_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by scoring epochs](/assets/experiments/014-montecarlo2-epochs.png)']
    lines += ['', 'A run uses one nominal seed; differences may reflect search order, randomness and '
              'unequal visit counts as well as scoring epochs.', '',
              '### What `MCTS_PROPAGATE_ROLLOUT_VALUE` means', '',
              '**A value of `False` does not turn off rollouts.** Phase 3 inherited `False` from the '
              'project configuration; the Phase 3 script did not override it. The saved run also records '
              '`MCTS_ROLLOUT_DEPTH=2` and a depth-weight decay of `0.5`.', '',
              'When an unvisited tree node is evaluated, `rollout()` scores its starting model, then '
              'scores after each randomly selected future action, up to the configured depth or until '
              'no actions are available. It returns their normalized weighted average, using weights '
              '`1`, `0.5`, and `0.25` when both future steps occur. This happens for either flag value. '
              'The saved results do not contain the individual actions or scores within each rollout.', '',
              'The flag controls what an internal tree node returns to its parent **after** receiving '
              'and adding a child result:', '',
              '- **`True`:** return the latest child result to the parent.',
              '- **`False`:** return the internal node’s accumulated value to the parent. Once a node '
              'has multiple visits, this can propagate prior results again, so ancestor values may not '
              'be simple averages of individual rollout scores.', '',
              'For example, if an internal node already has `0.4` and receives a new result of `0.6`, '
              'it adds the new result and holds `1.0`. With `True` it returns `0.6` upward; with '
              '`False` it returns `1.0`. This flag concerns propagation, not whether the `0.6` rollout ran.', '',
              'In the saved Phase 3 runs, each root action was visited once and the reported maximum '
              '**tree depth** was one. Each root-child score therefore came from its own rollout; the '
              'cumulative-return issue would matter when propagation passes through revisited internal '
              'nodes in a deeper tree. Tree depth and the number of random actions inside a rollout '
              'are different quantities.', '']
    return lines


def time_budget_section():
    plotted = []
    lines = ['', '## Phase 4: Monte Carlo 2 time budgets', '',
             'All rows use 20 scoring epochs. The baseline reuses the 20-epoch Phase 3 run. '
             'The other rows use nominal budgets of 20, 25 and 30 minutes; 5, 10 and 15 minutes '
             'were omitted because the baseline alone took approximately 17 minutes to complete '
             'its required initial evaluations. The budget is soft, so actual time can be longer.', '',
             'Cells show the root search value and visit count; **bold** marks the selected action. '
             'Values can reflect random future actions and the algorithm’s value propagation, '
             'and are not direct validation accuracies. Twenty epochs gave the largest A08 margin '
             'in the single Phase 3 run; that does not establish it is the best general setting.', '',
             '| Time budget | Actual time | Tree depth | Rollouts | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    paths = [('32 seconds (Phase 3)', RUN / 'montecarlo2_epochs/epochs_20.json')]
    paths += [(f'{minutes} minutes', RUN / 'montecarlo2_time' / f'{minutes}min' /
               'montecarlo2_epochs/epochs_20.json') for minutes in (20, 25, 30)]
    for label, path in paths:
        if not path.exists():
            lines.append(f'| {label} | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |')
            plotted.append((label, None))
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = {row['arm']: row for row in data['results']}
        plotted.append((label, rows))
        cells = []
        for i in range(1, 6):
            row = rows[f'candidate_{i}']
            cell = f"{row['score']:.4f} (n={row['visits']})" if row['score'] is not None else 'Not scored'
            cells.append(f'**{cell}**' if row['chosen'] else cell)
        lines.append(f"| {label} | {data['duration_sec'] / 60:.1f} min | {data['max_depth']} | "
                     f"{data['rollouts']} | " + ' | '.join(cells) + ' |')
    if any(rows is not None for _, rows in plotted):
        fig, ax = plt.subplots(figsize=(11, 5.5))
        positions = range(len(plotted))
        for i in range(1, 6):
            arm = f'candidate_{i}'
            ax.plot(positions, [rows[arm]['score'] if rows and rows[arm]['score'] is not None
                                else float('nan') for _, rows in plotted], marker='o',
                    linewidth=2.8 if i == 5 else 1.5, label=LABELS[arm])
        ax.set(xlabel='Nominal time budget', ylabel='Mean search value',
               title='Experiment 14: Monte Carlo 2 time budgets',
               xticks=list(positions), xticklabels=[label for label, _ in plotted])
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        TIME_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(TIME_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by time budget](/assets/experiments/014-montecarlo2-time.png)']
    return lines + ['', 'Each setting uses one nominal seed. Greater budgets may allow deeper exploration, '
                    'but do not guarantee equal visits or identical random trajectories.', '']


def full_growth_section():
    target = RUN / 'full_growth_5min'
    lines = ['', '## Phase 5: full growth from the Experiment 010 starter', '',
             'CIFAR-10, the Experiment 010 tiny starter (296 parameters), 10 generations of 20 '
             'training epochs, and seeds 1 and 2. This uses the Experiment 010 Pass 2 selected '
             'settings with Monte Carlo 2, **20 scoring epochs** and a **5-minute simulation '
             'budget**. The budget is soft and may be exceeded while the search completes its '
             'minimum evaluations.', '',
             'Other settings: UCB C=1.0, legacy UCB formula, rollout propagation enabled, '
             'rollout depth 1, minimum iterations 3, and a 10% simulation training subset. '
             'Rollout epochs and rollout learning rate follow the Experiment 010 selected values '
             'but Monte Carlo 2 does not use them for intermediate training. Its depth-weight decay '
             'is the current configured value, since Experiment 010 did not tune that parameter.', '',
             '| Seed | Status | Final validation | Final test | Best validation |',
             '| ---: | --- | ---: | ---: | ---: |']
    for seed in (1, 2):
        path = target / f'seed_{seed}' / 'run.json'
        if not path.exists():
            lines.append(f'| {seed} | Pending | — | — | — |')
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        lines.append(f"| {seed} | Complete | {100 * row['final_validation_accuracy']:.2f}% | "
                     f"{100 * row['final_test_accuracy']:.2f}% | {100 * row['best_validation_accuracy']:.2f}% |")
    return lines + ['', 'Experiment 010 selected settings in sequential sweeps; its report does not '
                    'provide a completed joint run of the final configuration as a matched baseline. '
                    'The proposed 40-generation confirmation was not started. Differences in the '
                    'algorithm and the two simulation-budget settings limit causal attribution.', '']


def main():
    summary = json.loads((RUN / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['results']
    fig, ax = plt.subplots(figsize=(11, 6))
    for row in rows:
        vals = row['history']['val_acc']
        ax.plot(range(81, 81 + len(vals)), [100 * v for v in vals], label=LABELS[row['arm']],
                linewidth=2.5 if row['original_chosen'] else 1.5)
    ax.set(xlabel='Total training epoch', ylabel='Held-out validation accuracy (%)',
           title='Experiment 14: training after each generation-3 action')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, dpi=150)
    plt.close(fig)
    lines = ['---', 'title: "Experiment 14: replaying the epoch-80 decision"', '---', '',
             'All arms start from the Experiment 12 generation-3 pre-decision checkpoint, after epoch 80. '
             'Each recorded action and an unchanged control is trained for 20 epochs with the same '
             'nominal seed, training recipe and saved learning-rate schedule. The primary comparison uses '
             '3,880 validation examples outside the original 1,120-example simulation validation subset.', '',
             'The five actions and the original simulation scores are from the recorded decision. '
             'Resetting the seed does not guarantee identical random draws across different architectures. '
             'This is one checkpoint and one training seed; it is not a multi-seed estimate.', '',
             '| Action | Original simulation | Parameters | Initial held-out | After 1 epoch | After 20 epochs | Difference vs control | Full validation after 20 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        pct = lambda x: f'{x * 100:.2f}%'
        score = pct(row['original_simulation_score']) if row['original_simulation_score'] is not None else '—'
        lines.append(f"| {LABELS[row['arm']]} | {score} | {row['parameter_count']:,} | "
                     f"{pct(row['initial_heldout_accuracy'])} | {pct(row['history']['val_acc'][0])} | "
                     f"{pct(row['final_heldout_accuracy'])} | {row['gain_vs_control_pp']:+.2f} pp | "
                     f"{pct(row['final_full_validation_accuracy'])} |")
    lines += ['', '![Held-out validation accuracy for all replay arms](/assets/experiments/014-action-replay-validation.png)', '',
              'Full validation includes the original simulation-validation subset; held-out validation is the primary '
              'comparison. No test-set scores were used. The original run followed a different random trajectory, '
              'so its subsequent accuracy need not be reproduced exactly.', '']
    lines += scoring_section()
    lines += montecarlo2_section()
    lines += time_budget_section()
    lines += full_growth_section()
    from documentation.website.scripts.generate_experiment_014_seed1_analysis import generate as generate_seed1
    lines += generate_seed1()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {REPORT} and {CHART}')


if __name__ == '__main__':
    main()
