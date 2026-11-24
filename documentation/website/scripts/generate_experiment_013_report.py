"""Create Experiment 013's report and learning-curve figure from saved results."""
import json
import re
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1'
REPORT = ROOT / 'documentation/website/content/experiments/experiment-013-monte-carlo-2.md'
DETAILS_HEADING = '## Phase details'
COMPARISON_REPORT_EPOCHS = 300  # Phase 18 charts show the first 300 epochs (10 generations)
SUMMARY_HEADING = '## Summary'
NEWLINE = chr(10)
CHART = ROOT / 'documentation/website/app/public/assets/experiments/018-action-replay-validation.png'
SCORING_CHART = ROOT / 'documentation/website/app/public/assets/experiments/018-scoring-epochs.png'
MCTS_CHART = ROOT / 'documentation/website/app/public/assets/experiments/018-montecarlo2-epochs.png'
TIME_CHART = ROOT / 'documentation/website/app/public/assets/experiments/018-montecarlo2-time.png'
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
               title='Experiment 013: effect of scoring epochs', xticks=epochs_list)
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
        lines += ['', '![Simulation accuracy by scoring epochs and action](/assets/experiments/018-scoring-epochs.png)']
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
               title='Experiment 013: Monte Carlo 2 search values', xticks=sorted(plotted))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        MCTS_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(MCTS_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by scoring epochs](/assets/experiments/018-montecarlo2-epochs.png)']
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
               title='Experiment 013: Monte Carlo 2 time budgets',
               xticks=list(positions), xticklabels=[label for label, _ in plotted])
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        TIME_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(TIME_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by time budget](/assets/experiments/018-montecarlo2-time.png)']
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


SCORE_FIX_LIST = [
    '- **True mean values.** Each tree node now passes the latest rollout value to its parent, so '
    '`value / visits` is a real mean. Previously, with `MCTS_PROPAGATE_ROLLOUT_VALUE=False`, nodes '
    'returned their accumulated sum and ancestors added sums of sums. Monte Carlo 2 now ignores that flag.',
    '- **Mean-based UCB.** The legacy (non-square-root) UCB formula now exploits the mean value '
    'instead of the raw sum, matching the final choice, which uses the mean.',
    '- **No root rollout.** The starting model is no longer rolled out before the search; its score '
    'was never used to choose an action. Covering N root actions now takes N rollouts instead of N + 1.',
    '- **Score memory.** Only the node\'s initial score is remembered for reuse; scores of random '
    'rollout steps were stored under keys that could never be reused.',
    '- **Board records.** Each candidate\'s recorded composite score is the weighted rollout value '
    'used by the search (not the last step\'s score), and the chosen row is marked by tree node rather '
    'than by its action text.',
]


CHECK_TABLE_HEADER = [
    '| Seed | Search time | Widen 256 | Widen 128 | Add linear | Conv without BN/ReLU | A08: conv + BN/ReLU | Selected |',
    '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']


def _check_table(runs):
    """Table rows for a score-fix check plus a one-line result about A08."""
    lines, a08_top, ties = [], 0, []
    for run in runs:
        rows = {row['arm']: row for row in run['results']}
        best = max(row['score'] for row in rows.values())
        cells, selected = [], '—'
        for i in range(1, 6):
            row = rows[f'candidate_{i}']
            cell = f"{row['score']:.4f} (n={row['visits']})"
            cells.append(f'**{cell}**' if row['chosen'] else cell)
            if row['chosen']:
                selected = LABELS[f'candidate_{i}']
        tied = [LABELS[arm] for arm, row in rows.items() if abs(row['score'] - best) < 1e-9]
        if LABELS['candidate_5'] in tied:
            a08_top += 1
        if len(tied) > 1:
            ties.append(f"seed {run['seed']} ({', '.join(tied)})")
        lines.append(f"| {run['seed']} | {run['duration_sec'] / 60:.1f} min | " + ' | '.join(cells)
                     + f' | {selected} |')
    result = f'**Result:** A08 had the highest score, alone or tied, in {a08_top} of {len(runs)} seeds.'
    if ties:
        result += (' Exact ties occurred in ' + '; '.join(ties) + '; the search keeps the first tied '
                   'candidate in action order, so a tie can select another action.')
    return lines, result


def score_fix_section():
    path = RUN / 'score_fix_check/summary.json'
    lines = ['', '## Phase 6: replaying the A08 decision with the fixed Monte Carlo 2', '',
             'Phase 5 did not reproduce an A08-like jump. Two causes were found: convolution widening '
             'was not enabled in Phase 5, so every added convolution stayed at 8 → 8 channels and no '
             'action comparable to A08 (a 64 → 64 convolution, +37k parameters) existed; and every '
             'candidate received exactly one visit, so the search reduced to "score each action once" '
             'with near-tied scores from generation 3 onwards.', '',
             'The following issues were fixed in the Monte Carlo 2 simulation algorithm '
             '(`montecarlo_alg_2.py`):', '',
             *SCORE_FIX_LIST, '',
             'This phase repeats the Phase 3 search on the same generation-3 checkpoint '
             'and five recorded root actions, with the fixed algorithm and the settings planned for '
             'Phase 7: **rollout depth 0**, **5 scoring epochs**, convolution widening enabled, UCB C=1.0, '
             'minimum iterations 3 and a 32-second soft budget. Three search seeds are run to show '
             'how stable the ranking is. Cells show the mean search value and visit count; **bold** '
             'marks the selected action.', '',
             *CHECK_TABLE_HEADER]
    if not path.exists():
        lines.append('| — | Pending | Pending | Pending | Pending | Pending | Pending | Pending |')
        return lines + ['', 'Run the experiment script with `--phase check-score-fix`, then regenerate '
                        'this report.', '']
    runs = json.loads(path.read_text(encoding='utf-8'))['runs']
    rows, result = _check_table(runs)
    lines += rows
    return lines + ['', result, '',
                    'Each search scores every root action once (one rollout per action, no random follow-up '
                    'actions), so one search costs one 5-epoch scoring run per candidate instead of two or '
                    'three 20-epoch runs in Phase 5. Scores are single measurements on the 1,120-example '
                    'simulation-validation subset and remain noisy below the top candidate.', '']


def _spread(runs, arm):
    scores = [100 * next(row['score'] for row in run['results'] if row['arm'] == arm) for run in runs]
    return f'{sum(scores) / len(scores):.2f}% ({min(scores):.2f}–{max(scores):.2f})'


def _a08_leads(runs):
    leads = []
    for run in runs:
        scores = {row['arm']: row['score'] for row in run['results']}
        a08 = scores.pop('candidate_5')
        leads.append(f"{100 * (a08 - max(scores.values())):+.1f}")
    return ', '.join(leads) + ' pp'


GRID_CELLS = [(0.05, 'moderate'), (0.05, 'ccs'), (1.0, 'moderate'), (1.0, 'ccs')]
GRID_SET_NAMES = {'moderate': 'Moderate', 'ccs': 'CCS'}


def _grid_runs():
    """{(C, selector): {seed: run.json plus total search minutes and final parameters}}."""
    runs = {}
    for ucb_c, selector in GRID_CELLS:
        cell = RUN / 'full_grid' / f"c{str(ucb_c).replace('.', 'p')}_{selector}"
        for seed in (1, 2):
            path = cell / f'seed_{seed}' / 'run.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text(encoding='utf-8'))
            board = cell / f'seed_{seed}' / 'board'
            searches = [json.loads(p.read_text(encoding='utf-8'))
                        for p in (board / 'simulations').glob('simulation_gen_*.json')]
            metrics = json.loads((board / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']
            row['search_minutes'] = sum(search['durationSec'] for search in searches) / 60
            row['final_parameters'] = metrics[-1]['paramCount']
            runs.setdefault((ucb_c, selector), {})[seed] = row
    return runs


def _near_ties(search):
    best = max(row['score'] for row in search['candidates'])
    return sum(best - row['score'] < 0.01 for row in search['candidates'])


def sim_fixes_growth_section():
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    target = RUN / 'full_growth_sim_fixes'
    lines = ['', '## Phase 12: full growth with the simulation fixes', '',
             'The best Phase 10 cell (UCB C = 1.0, CCS simulation set) repeated with all four simulation fixes made '
             'after Phase 10: lazy expansion, the restarted scoring learning rate, futures that continue from '
             'trained models, and measuring each node twice before expanding it. Everything else matches Phase 10: '
             'the Experiment 010 starter, 10 generations of 20 epochs, 5 scoring epochs with the mean of the last '
             'two, rollout depth 0, convolution widening including fan-out, and 24 seconds of search per candidate '
             '(at least 5 minutes). Seeds 1 and 2.', '',
             '| Run | Seed | Final validation | Final test | Final parameters | Search time |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    runs = {seed: _load_growth_run(target / f'seed_{seed}') for seed in (1, 2)
            if (target / f'seed_{seed}' / 'run.json').exists()}
    for seed in (1, 2):
        if seed not in runs:
            lines.append(f'| Phase 12 | {seed} | Pending | — | — | — |')
            continue
        metrics, searches, run = runs[seed]
        lines.append(f"| **Phase 12** | {seed} | **{100 * run['final_validation_accuracy']:.2f}%** | "
                     f"{100 * run['final_test_accuracy']:.2f}% | {metrics[-1]['paramCount']:,} | "
                     f"{sum(search['durationSec'] for search in searches) / 60:.0f} min |")
    ref_metrics, ref_searches, _ = _load_growth_run(RUN / 'full_grid' / 'c1p0_ccs' / 'seed_1')
    for label, directory in (('Phase 10, C = 1.0, CCS', RUN / 'full_grid' / 'c1p0_ccs' / 'seed_1'),
                             ('Phase 5 (no widening, 20 scoring epochs)', RUN / 'full_growth_5min' / 'seed_1'),
                             ('the origin run (family search)', EXP13_RUN)):
        metrics, searches, run = _load_growth_run(directory)
        search_time = '—' if directory == EXP13_RUN else f"{sum(s['durationSec'] for s in searches) / 60:.0f} min"
        lines.append(f"| {label} | 1 | {100 * run['final_validation_accuracy']:.2f}% | "
                     f"{100 * run['final_test_accuracy']:.2f}% | {metrics[-1]['paramCount']:,} | {search_time} |")
    lines += ['', "The origin run's search time is not shown: its simulation files record only part of the family "
              'search.']
    if not runs:
        return lines + ['', 'Run with `--phase full-sim-fixes`, then regenerate this report.', '']

    fig, (acc, size) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    curves = [(f'Phase 10, C = 1.0, CCS, seed 1', ref_metrics, '#888888')]
    curves += [(f'Phase 12, seed {seed}', runs[seed][0], color) for seed, color in zip(runs, ('#2a78d6', '#eb6834'))]
    for label, metrics, color in curves:
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], color=color, lw=1.6,
                 label=f"{label} validation (final {100 * metrics[-1]['valAcc']:.1f}%)")
        acc.plot(epochs, [100 * row['trainAcc'] for row in metrics], color=color, lw=1.1, ls='--',
                 label=f'{label} training')
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', color=color, lw=1.6, label=label)
    acc.set(ylabel='Accuracy (%)', ylim=(0, 80), title='Phase 12 against the best Phase 10 cell')
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    for axis in (acc, size):
        axis.grid(alpha=.25)
        for boundary in range(20, 200, 20):
            axis.axvline(boundary + .5, color='#bbbbbb', ls=':', lw=.8)
    acc.legend(loc='lower right', fontsize=8)
    size.legend(loc='upper left', fontsize=8)
    fig.savefig(ASSET_DIR / '018-sim-fixes-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    lines += ['', '![Phase 12 against the best Phase 10 cell](/assets/experiments/018-sim-fixes-training.png)', '',
              'Solid lines are validation accuracy, dashed lines training accuracy.', '']

    columns = {'Phase 10, C = 1.0, CCS': (ref_metrics, ref_searches),
               **{f'Phase 12 seed {seed}': runs[seed][:2] for seed in runs}}
    generations = range(max(len(searches) for _, searches in columns.values()))
    header = ['| Generation | ' + ' | '.join(columns) + ' |', '| ---: |' + ' ---: |' * len(columns)]

    def table(cell):
        rows = []
        for generation in generations:
            cells = []
            for metrics, searches in columns.values():
                end = {row['globalEpoch'] // 20: row['valAcc'] for row in metrics}
                cells.append(cell(searches[generation], end, generation) if generation < len(searches) else '—')
            rows.append(f'| {generation} | ' + ' | '.join(cells) + ' |')
        return rows

    lines += ['### Did the candidates separate?', '',
              'Number of candidates within 1 pp of the best candidate in each search (fewer means the search could '
              'tell the leader apart).', '', *header]
    lines += table(lambda search, end, generation: f"{_near_ties(search)} of {len(search['candidates'])}")
    lines += ['', '### Best candidate score against the model being grown', '',
              'Score of the selected candidate, and the validation accuracy of the model when the search ran '
              '(in brackets).', '', *header]
    lines += table(lambda search, end, generation:
                   f"{100 * search['scoreChosen']:.1f}% ({100 * end[generation]:.1f}%)")
    lines += ['', '### Decisions', '',
              'Selected action type per generation, with the change in validation accuracy over the next '
              'generation.', '', '| Generation | ' + ' | '.join(columns) + ' |', '| ---: |' + ' --- |' * len(columns)]
    lines += table(lambda search, end, generation: f"{group_key({'action': search['actionChosen']})} "
                                                   f"({100 * (end[generation + 1] - end[generation]):+.1f} pp)")
    return lines + _sim_fixes_growth_findings(columns)


def _sim_fixes_growth_findings(columns):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    mean = lambda values: sum(values) / len(values)
    late = range(5, 9)
    phase12 = {name: value for name, value in columns.items() if name.startswith('Phase 12')}
    if len(phase12) < 2:
        return ['']
    finals = [100 * metrics[-1]['valAcc'] for metrics, _ in phase12.values()]
    ref_metrics, ref_searches = columns['Phase 10, C = 1.0, CCS']
    late_ties = {name: mean([_near_ties(searches[g]) for g in late]) for name, (_, searches) in columns.items()}

    def deficit(metrics, searches, generations):
        end = {row['globalEpoch'] // 20: row['valAcc'] for row in metrics}
        return mean([100 * (end[g] - searches[g]['scoreChosen']) for g in generations])

    widen = sum(group_key({'action': search['actionChosen']}) == 'Widen convolution'
                for _, searches in phase12.values() for search in searches)
    seed1_metrics, seed1_searches = phase12['Phase 12 seed 1']
    seed1_end = {row['globalEpoch'] // 20: row['valAcc'] for row in seed1_metrics}
    seed1_dropout = [g for g, search in enumerate(seed1_searches)
                     if group_key({'action': search['actionChosen']}) == 'Dropout']
    after_gen3 = {name: 100 * (metrics[-1]['valAcc'] - {row['globalEpoch'] // 20: row['valAcc']
                                                        for row in metrics}[3])
                  for name, (metrics, _) in columns.items()}
    largest_step = max(row['paramCount'] - search['paramCountBefore']
                       for _, searches in columns.values() for search in searches[5:]
                       for row in search['candidates'] if group_key(row) != 'Widen convolution')
    seed2_metrics = phase12['Phase 12 seed 2'][0]
    return ['', '### What Phase 12 shows', '',
            '- **Growth slows after generation 3 in every run.** From the end of generation 3 to the end, '
            'validation accuracy rose by '
            + ', '.join(f'{gain:.1f} pp ({name})' for name, gain in after_gen3.items()) + ' over six '
            f'generations. In generations 5–8 no candidate except widening added more than {largest_step} '
            'parameters: every new convolution is 8→8 channels, because new layers take the channel count of the '
            'layers they connect, and widening doubles a single layer from 8 to 16 channels. Training accuracy '
            f'stays at validation accuracy (seed 2 ended at {100 * seed2_metrics[-1]["trainAcc"]:.1f}% training, '
            f'{100 * seed2_metrics[-1]["valAcc"]:.1f}% validation), so the network is too small rather than '
            'over-fitting. The Phase 1 action that gave +18.6 pp was a 64-channel convolution; no action of that '
            'size is on offer here, so even a perfect search could only pick between small steps.',
            f'- **Final accuracy did not improve.** The two seeds ended at {finals[0]:.2f}% and {finals[1]:.2f}% '
            f'validation (mean {mean(finals):.2f}%), against {100 * ref_metrics[-1]["valAcc"]:.2f}% for the same '
            'cell in Phase 10. The two seeds differ by more than Phase 12 differs from Phase 10, and the four '
            'Phase 10 cells (one seed each) ranged from 59.38% to 63.06%, so the runs show no change in either '
            'direction.',
            f'- **Late-generation candidates now separate.** In generations 5–8, Phase 10 had on average '
            f'{late_ties["Phase 10, C = 1.0, CCS"]:.1f} candidates within 1 pp of the best (up to 34 of 77 in '
            'generation 7); Phase 12 had '
            + ' and '.join(f'{late_ties[name]:.1f}' for name in phase12) + '. This is what the fixes were meant to '
            'do, but the better separation did not show up in the final accuracy.',
            f'- **Scores now sit below the model being grown.** With the restarted learning rate, scoring trains '
            f'each candidate for 5 epochs at about 0.1 while the main training in generations 5–8 runs at about 0.035 '
            f'down to 0.001. In generations 6–8 the selected candidate scored on average '
            + ' and '.join(f'{deficit(*phase12[name], range(6, 9)):.1f}' for name in phase12)
            + f' pp below the model\'s validation accuracy, against {deficit(ref_metrics, ref_searches, range(6, 9)):.1f} '
            'pp in Phase 10. The score now measures how a candidate does after a short high-learning-rate restart, '
            'and the main training then continues the low-learning-rate schedule instead. Whether the two rank '
            'candidates differently was not measured.',
            f'- **Seed 1 spent its late generations on Dropout.** It selected Dropout in generations '
            + ' and '.join(map(str, seed1_dropout)) + ', which added no parameters and left validation accuracy '
            'flat (' + ' and '.join(f'{100 * seed1_end[g + 1]:.1f}%' for g in seed1_dropout) + ' after them). '
            'Dropout leading under high-learning-rate scoring fits the point above, but the cause was not measured.',
            f'- **Widening was never selected** ({widen} of {sum(len(s) for _, s in phase12.values())} decisions), '
            'although it was runner-up in several searches; the runs ended at 4,400–5,600 parameters, '
            'far below the origin run\'s 80,887.',
            '- **Searches went shallower.** Measuring each node twice uses rollouts that used to expand the tree: '
            'Phase 12 searches reached depth 1 from generation 1 on, Phase 10 reached depth 2. Search time was '
            'unchanged (137 and 156 min against 157 min).', '',
            '### What to change', '',
            '- **Offer larger growth steps.** Let new convolutions choose a wider output (for example 16, 32 or 64 '
            'channels, as in the Phase 1 action A08), or widen a whole chain of layers at once instead of one '
            'layer. This is a change to the actions, not to the simulation.',
            '- **Keep the main learning-rate schedule.** Experiment 008 showed that one global schedule beats '
            'restarting at 0.1 every generation. The scoring restart stays a simulation-only choice; whether it '
            'ranks candidates the same way as the main schedule would is still open.',
            f'- **Run more seeds before comparing cells.** The two Phase 12 seeds differ by '
            f'{abs(finals[1] - finals[0]):.1f} pp; the four Phase 10 cells, one seed each, spread over 3.7 pp.', '']


def _ranks(values):
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks = [0] * len(values)
    for rank, index in enumerate(order, 1):
        ranks[index] = rank
    return ranks


def decision_replay_section():
    path = RUN / 'decision_replay/gen_3/summary.json'
    lines = ['', '## Phase 13: every growing candidate of one late decision, trained for real', '',
             'Why do generations after the third bring so little? Phase 13 takes the Phase 12 seed-2 search after '
             'generation 3 and trains every candidate that grows the network (all except Dropout, Delete layer and '
             'Delete neurons: 16 of 27) for a full 20-epoch generation on the full training set with the main '
             'learning-rate schedule, plus the unchanged model as a control. This shows how far the selected '
             'action was from the best one available, and whether any available action could have given a large '
             'gain.', '',
             'Phase 12 saved no checkpoints, so the generation-3 model is grown again from the starter with the '
             "run's recorded choices and training recipe. It has the same architecture but not the same weights, "
             'because the original searches consumed random numbers between generations.', '']
    if not path.exists():
        return lines + ['Pending. Run with `--phase decision-replay`.', '']
    summary = json.loads(path.read_text(encoding='utf-8'))
    control, *arms = summary['results']
    base = control['final_validation_accuracy']
    real = [arm['final_validation_accuracy'] for arm in arms]
    simulated = [arm['simulation_score'] for arm in arms]
    real_ranks, sim_ranks = _ranks(real), _ranks(simulated)
    lines += [f"The rebuilt model reached {100 * summary['rebuilt_validation_accuracy']:.2f}% validation at the "
              'end of generation 3 (Phase 12 seed 2: 53.20%). The control trained one more generation without an '
              f'action and reached {100 * base:.2f}%.', '',
              '| Real rank | Action | Real validation | Gain over control | Simulation score | Simulation rank |',
              '| ---: | --- | ---: | ---: | ---: | ---: |']
    for index in sorted(range(len(arms)), key=lambda index: real_ranks[index]):
        arm = arms[index]
        label = _action_label(arm['action']) + (' **(selected)**' if arm['chosen'] else '')
        lines.append(f"| {real_ranks[index]} | {label} | {100 * real[index]:.2f}% | "
                     f"{100 * (real[index] - base):+.2f} pp | {simulated[index]:.4f} | {sim_ranks[index]} |")
    chosen = next(index for index, arm in enumerate(arms) if arm['chosen'])
    best = real_ranks.index(1)
    count = len(arms)
    spearman = 1 - 6 * sum((a - b) ** 2 for a, b in zip(real_ranks, sim_ranks)) / (count * (count ** 2 - 1))
    gains = sorted(100 * (value - base) for value in real)
    lines += _decision_replay_histories(control, arms, real_ranks, sim_ranks)
    return lines + ['', '### What Phase 13 shows', '',
                    f'- **The selected action ranked {real_ranks[chosen]} of {count} in real training**, '
                    f'{100 * (real[best] - real[chosen]):.2f} pp below the best candidate '
                    f'({_action_label(arms[best]["action"])}).',
                    f'- **The best available gain over the control was {gains[-1]:+.2f} pp**; the middle candidate '
                    f'gained {gains[count // 2]:+.2f} pp and the worst {gains[0]:+.2f} pp.',
                    f'- **Simulation and real ranking agree with a rank correlation of {spearman:.2f}** '
                    '(1 is the same order, 0 no relation).', '']


def _decision_replay_histories(control, arms, real_ranks, sim_ranks):
    """Training curves of every Phase 13 arm, all started from the same generation-3 model."""
    chosen = next(index for index, arm in enumerate(arms) if arm['chosen'])
    highlighted = {}
    for index, color, name in ((sim_ranks.index(1), '#1baa7a', 'Simulation best'),
                               (chosen, '#eb6834', 'Selected by the search'),
                               (real_ranks.index(1), '#2a78d6', 'Real best')):
        previous = highlighted.get(index, (None, ''))[1]
        highlighted[index] = (color, f'{name} + {previous}' if previous else name)
    base_val = control['history']['val_acc']
    epochs = range(1, len(base_val) + 1)
    fig, (val, gap, train) = plt.subplots(3, 1, figsize=(11, 11), sharex=True, layout='constrained',
                                          gridspec_kw={'height_ratios': [2, 2, 1.5]})
    order = [index for index in range(len(arms)) if index not in highlighted] + list(highlighted)
    for index in order:
        history = arms[index]['history']
        color, role = highlighted.get(index, ('#b8b8b8', None))
        width, layer = (2.0, 3) if role else (0.9, 1)
        label = (f"{role}: {_short_label(arms[index]['action'])} "
                 f"({100 * history['val_acc'][-1]:.1f}%)") if role else None
        val.plot(epochs, [100 * value for value in history['val_acc']], color=color, lw=width, zorder=layer,
                 label=label)
        gap.plot(epochs, [100 * (value - base) for value, base in zip(history['val_acc'], base_val)],
                 color=color, lw=width, zorder=layer)
        train.plot(epochs, history['train_loss'], color=color, lw=width, zorder=layer)
    for axis, values in ((val, [100 * value for value in base_val]), (train, control['history']['train_loss'])):
        axis.plot(epochs, values, color='#222222', lw=2.0, ls='--', zorder=4,
                  label=f"No action ({100 * base_val[-1]:.1f}%)" if axis is val else None)
    gap.axhline(0, color='#222222', lw=2.0, ls='--', zorder=4)
    val.plot([], [], color='#b8b8b8', lw=0.9, label='Other candidates')
    val.set(ylabel='Validation accuracy (%)',
            title='Phase 13: every candidate trained for 20 epochs from the same generation-3 model')
    gap.set(ylabel='Validation minus no action (pp)')
    train.set(xlabel='Epoch after the action', ylabel='Training loss')
    for axis in (val, gap, train):
        axis.grid(alpha=.25)
    val.legend(loc='lower right', fontsize=8)
    fig.savefig(ASSET_DIR / '018-decision-replay-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    note = ' One candidate holds more than one role.' if len(highlighted) < 3 else ''
    return ['', '![Phase 13 training histories](/assets/experiments/018-decision-replay-training.png)', '',
            'Every line starts from the same model and training seed and differs only in the action applied '
            'before epoch 1. The dashed black line is the control without an action. The middle panel '
            'subtracts the control, so a line above zero is ahead of it at that epoch.' + note, '']


def _short_label(action):
    return _action_label(action).replace(' — ', ': ').split(' / ')[0]


def _partial_growth_run(directory):
    """Per-epoch metrics and searches of a seed that stopped before writing run.json."""
    board = directory / 'board'
    metrics = json.loads((board / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']
    searches = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
        (board / 'simulations').glob('simulation_gen_*.json'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))]
    return metrics, searches, None


def _widening_training_chart(runs):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    short = {'Widen convolution': 'W', 'Sequential convolution + BN': 'SC', 'Sequential convolution without BN':
             'SC-', 'Residual convolution': 'RC', 'Sequential linear': 'SL', 'Add neurons': 'N',
             'Delete neurons': 'DN', 'Dropout': 'D', 'Delete layer': 'DL', 'Other': 'O'}
    references = []  # Phase 14 only; Phase 12 is compared in the text
    colors = {1: '#2a78d6', 2: '#eb6834', 3: '#1baa7a'}
    fig, (acc, size) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout='constrained',
                                    gridspec_kw={'height_ratios': [2, 1.3]})
    curves = references + [(f'Phase 14 seed {seed}' + ('' if info else ' (stopped)'), metrics, colors[seed])
                           for seed, (metrics, _, info) in runs.items()]
    for label, metrics, color in curves:
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        width = 1.8 if label.startswith('Phase 14') else 1.2
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], color=color, lw=width,
                 label=f"{label} validation (last {100 * metrics[-1]['valAcc']:.1f}%)")
        acc.plot(epochs, [100 * row['trainAcc'] for row in metrics], color=color, lw=width * .7, ls='--')
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', color=color, lw=width, label=label)
    for offset, (seed, (metrics, searches, _)) in enumerate(runs.items()):
        for generation, search in enumerate(searches):
            kind = group_key({'action': search['actionChosen']})
            size.annotate(short.get(kind, kind[:3]), (20 * (generation + 1) + .5, 0), xycoords=('data', 'axes fraction'),
                          xytext=(3, 4 + 11 * offset), textcoords='offset points', fontsize=7, color=colors[seed])
    acc.plot([], [], color='#666666', lw=1, ls='--', label='Dashed: training accuracy')
    acc.set(ylabel='Accuracy (%)', ylim=(0, 80), title='Phase 14: full growth with widening offered for every convolution')
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    for axis in (acc, size):
        axis.grid(alpha=.25)
        for boundary in range(20, 200, 20):
            axis.axvline(boundary + .5, color='#bbbbbb', ls=':', lw=.8)
    acc.legend(loc='lower right', fontsize=8)
    size.legend(loc='upper left', fontsize=8)
    fig.savefig(ASSET_DIR / '018-widening-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return ['', '![Phase 14 training history](/assets/experiments/018-widening-training.png)', '',
            'Solid lines are validation accuracy, dashed lines training accuracy; dotted vertical lines mark '
            'generation boundaries, where the selected action is applied. The letters at the bottom name the '
            'action applied at each boundary (W widen convolution, SC sequential convolution with BN + ReLU, '
            'SC- without them, RC residual convolution, SL sequential linear, N add neurons, O other, e.g. a '
            'residual linear layer). Parameters are in thousands.', '']


WIDENING_SEEDS = (1, 2, 3)


def widening_growth_section():
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    target = RUN / 'full_growth_widening'
    lines = ['', '## Phase 14: full growth with widening for every convolution', '',
             'Phase 12 repeated after the widening refactor (`91085d0`). Before it, widening was offered only '
             'where copied channels kept the output identical: in practice only the first convolution, so 1–2 '
             'of the 27–90 candidates per search. Now every convolution can be widened ×2 or ×4. The new '
             'width is carried through BatchNorm, residual sums and the classifier after global pooling. New '
             'channels start with random weights, and every layer that reads them starts with zero weights for '
             'them, so the output is unchanged after the action. Everything else matches Phase 12. Seeds 1–3.', '',
             '| Run | Seed | Final validation | Final test | Final parameters | Search time |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    runs = {}
    for seed in WIDENING_SEEDS:
        directory = target / f'seed_{seed}'
        if (directory / 'run.json').exists():
            runs[seed] = _load_growth_run(directory)
        elif (directory / 'board/metrics/training.json').exists():
            runs[seed] = _partial_growth_run(directory)
    rows = [('**Phase 14**', seed, runs.get(seed)) for seed in WIDENING_SEEDS]
    rows += [('Phase 12', seed, _load_growth_run(RUN / 'full_growth_sim_fixes' / f'seed_{seed}')) for seed in (1, 2)]
    for label, seed, run in rows:
        if run is None:
            lines.append(f'| {label} | {seed} | Not run | — | — | — |')
            continue
        metrics, searches, info = run
        if info is None:
            lines.append(f"| {label} | {seed} | Stopped after generation {metrics[-1]['generation']}: "
                         f"{100 * metrics[-1]['valAcc']:.2f}% | — | {metrics[-1]['paramCount']:,} | "
                         f"{sum(search['durationSec'] for search in searches) / 60:.0f} min |")
            continue
        lines.append(f"| {label} | {seed} | {100 * info['final_validation_accuracy']:.2f}% | "
                     f"{100 * info['final_test_accuracy']:.2f}% | {metrics[-1]['paramCount']:,} | "
                     f"{sum(search['durationSec'] for search in searches) / 60:.0f} min |")
    if not runs:
        return lines + ['', 'Run with `--phase full-widening --seeds 1 2 3`, then regenerate this report.', '']
    lines += _widening_training_chart(runs)

    def widening(search):
        return [row for row in search['candidates'] if group_key(row) == 'Widen convolution']

    lines += ['', '### How often widening was offered and selected', '',
              '| Run | Seed | Widening candidates per search | Widening share of candidates | Times selected |',
              '| --- | ---: | ---: | ---: | ---: |']
    for label, seed, run in rows:
        if run is None:
            continue
        searches = run[1]
        offered = sum(len(widening(search)) for search in searches)
        total = sum(len(search['candidates']) for search in searches)
        chosen = sum(group_key({'action': search['actionChosen']}) == 'Widen convolution' for search in searches)
        lines.append(f'| {label} | {seed} | {offered / len(searches):.1f} | {100 * offered / total:.0f}% | '
                     f'{chosen} of {len(searches)} |')

    lines += ['', '### Decisions', '',
              'Selected action type per generation, with parameters after the action and the change in '
              'validation accuracy over the next generation.', '',
              '| Generation | ' + ' | '.join(f'Phase 14 seed {seed}' for seed in runs) + ' |',
              '| ---: |' + ' --- |' * len(runs)]
    for generation in range(max(len(run[1]) for run in runs.values())):
        cells = []
        for metrics, searches, _ in runs.values():
            end = {row['globalEpoch'] // 20: row for row in metrics}
            if generation + 1 not in end:
                cells.append('—')
                continue
            before, after = end[generation], end[generation + 1]
            cells.append(f"{group_key({'action': searches[generation]['actionChosen']})} "
                         f"({after['paramCount']:,}, {100 * (after['valAcc'] - before['valAcc']):+.1f} pp)")
        lines.append(f'| {generation} | ' + ' | '.join(cells) + ' |')
    return lines + ['']


NORMALIZATION_GENERATIONS = 5
NORMALIZATION_PARTS = (('16a', 'full_growth_normalization', 'Exp 010 starter, normalization action', '#c2357a'),
                       ('16b', 'full_growth_normalized_stem', 'Starter with BatchNorm after conv1', '#2a78d6'),
                       ('16c', 'full_growth_residual_stem', 'Starter with BatchNorm and a residual block', '#1baa7a'))


def _normalization_searches(searches):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    rows = []
    for search in searches:
        measured = [row for row in search['candidates'] if row.get('visits', 0) > 0]
        ordered = sorted(measured, key=lambda row: -row['score'])
        norms = [row for row in ordered if group_key(row) == 'Add normalization']
        best = norms[0] if norms else None
        rows.append({'generation': search['generation'], 'candidates': len(measured), 'norms': len(norms),
                     'best_norm': best, 'best_norm_rank': ordered.index(best) + 1 if best else None,
                     'best': ordered[0] if ordered else None,
                     'chosen': group_key({'action': search['actionChosen']}),
                     'norm_visits': sum(row['visits'] for row in norms) / len(norms) if norms else None,
                     'scores': [(row['score'], group_key(row) == 'Add normalization', bool(row.get('chosen')))
                                for row in measured]})
    return rows


def normalization_section():
    """Phase 16: the add-normalization action (a), and the starter with BatchNorm in its stem (b)."""
    lines = ['', '## Phase 16: is BatchNorm or the residual block the missing piece?', '',
             'The residual starter of the other branch (conv 16 → BatchNorm → ReLU, then relu(x + BatchNorm(conv(x)))) '
             'trained much better than our Experiment 010 starter, and our actions could not build two of its parts: '
             'a BatchNorm after an existing layer, and a residual block with BatchNorm. Phase 16 checks both. '
             'All parts are short Phase 14 runs (seed 1, CIFAR-10, the same search and scoring, '
             f'{NORMALIZATION_GENERATIONS} generations):', '',
             '- **16a**: our starter with the new add-normalization action (`541cab3`) among the candidates: a '
             'BatchNorm after any convolution or linear layer that has none, with default initialization. Does the '
             'simulation score it high?',
             '- **16b**: the same run from our starter with a BatchNorm already between conv1 and its ReLU, as in the '
             'residual starter\'s stem, with the same 8 channels and no residual block. If this run is much better '
             'than Phase 14, BatchNorm was what was missing; if not, the residual connection is needed as well.',
             '- **16c**: the 16b starter plus the residual block of the other branch\'s starter after the stem, '
             'relu(x + BatchNorm(conv(x))), still 8 channels. Compared with 16b, it shows what the residual block '
             'adds.', '']
    runs = {}
    for part, folder, label, color in NORMALIZATION_PARTS:
        directory = RUN / folder / 'seed_1'
        if not (directory / 'board/metrics/training.json').exists():
            phase = 'full-residual-stem' if part == '16c' else 'full-normalization'
            lines += [f'**{part}**: pending. Run with `--phase {phase} --seeds 1`.', '']
            continue
        complete = (directory / 'run.json').exists()
        metrics, searches, run = _load_growth_run(directory) if complete else _partial_growth_run(directory)
        runs[part] = (metrics, searches, run, label, color)
        finished = len({row['generation'] for row in metrics if row['epochInGeneration'] == 19})
        lines += [f'### {part}: {label}', '']
        if not complete:
            lines += [f'Partial: {finished} of {NORMALIZATION_GENERATIONS} generations trained and {len(searches)} '
                      'searches finished; read from the run files.', '']
        rows = _normalization_searches(searches)
        lines += ['| Search after generation | Candidates | Normalization candidates | Best normalization score '
                  '(rank) | Best score | Mean visits, normalization | Selected |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for row in rows:
            best_norm = (f"{100 * row['best_norm']['score']:.1f}% ({row['best_norm_rank']} of {row['candidates']})"
                         if row['best_norm'] else '—')
            best = f"{100 * row['best']['score']:.1f}%" if row['best'] else '—'
            visits = f"{row['norm_visits']:.1f}" if row['norm_visits'] is not None else '—'
            chosen = f"**{row['chosen']}**" if row['chosen'] == 'Add normalization' else row['chosen']
            lines.append(f"| {row['generation']} | {row['candidates']} | {row['norms']} | {best_norm} | {best} | "
                         f'{visits} | {chosen} |')
        if rows:
            jitter = random.Random(0)
            fig, axis = plt.subplots(figsize=(10, 4.5), layout='constrained')
            for row in rows:
                for score, is_norm, chosen in row['scores']:
                    axis.scatter(row['generation'] + jitter.uniform(-.25, .25), 100 * score,
                                 s=60 if is_norm else 22, color='#c2357a' if is_norm else '#b8b8b8',
                                 zorder=3 if is_norm else 2, edgecolors='#222222' if chosen else 'none',
                                 linewidths=1.6)
            axis.scatter([], [], s=60, color='#c2357a', label='Add normalization')
            axis.scatter([], [], s=22, color='#b8b8b8', label='Other candidates')
            axis.scatter([], [], s=40, color='white', edgecolors='#222222', linewidths=1.6, label='Selected')
            axis.set(xlabel='Search after generation', ylabel='Simulation score (%)',
                     title=f'Phase {part}: simulation scores of every candidate',
                     xticks=[row['generation'] for row in rows])
            axis.grid(alpha=.25)
            axis.legend(loc='lower right', fontsize=8)
            fig.savefig(ASSET_DIR / f'018-normalization-{part}-scores.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            lines += ['', f'![Phase {part} candidate scores](/assets/experiments/018-normalization-{part}-scores.png)',
                      '', 'Each dot is one measured candidate of one search; normalization candidates are magenta and '
                      'the selected candidate has a black ring.', '']
    if not runs:
        return lines
    reference = _load_growth_run(RUN / 'full_growth_widening' / 'seed_1')[0]
    longest = max(len(metrics) for metrics, *_ in runs.values())
    curves = [('Phase 14 seed 1', reference[:longest], '#9a9a9a')]
    curves += [(f'Phase {part}: {label}', metrics, color) for part, (metrics, _, _, label, color) in runs.items()]
    fig, axis = plt.subplots(figsize=(10, 4.5), layout='constrained')
    for label, curve, color in curves:
        epochs = [row['globalEpoch'] + 1 for row in curve]
        axis.plot(epochs, [100 * row['valAcc'] for row in curve], color=color, lw=1.8,
                  label=f"{label} (last {100 * curve[-1]['valAcc']:.1f}%)")
        axis.plot(epochs, [100 * row['trainAcc'] for row in curve], color=color, lw=1.1, ls='--')
    for boundary in range(20, 20 * NORMALIZATION_GENERATIONS, 20):
        axis.axvline(boundary + .5, color='#bbbbbb', ls=':', lw=.8)
    axis.set(xlabel='Training epoch', ylabel='Accuracy (%)', title='Phase 16 against the same epochs of Phase 14')
    axis.grid(alpha=.25)
    axis.legend(loc='lower right', fontsize=8)
    fig.savefig(ASSET_DIR / '018-normalization-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    lines += ['### Training', '', '![Phase 16 training history](/assets/experiments/018-normalization-training.png)', '',
              'Solid lines are validation accuracy, dashed lines training accuracy.', '',
              '| Run | Epochs | Validation at the last epoch | Test | Parameters |', '| --- | ---: | ---: | ---: | ---: |']
    for label, curve, _ in curves:
        complete = next((run for part, (metrics, _, run, *_) in runs.items() if metrics is curve), None)
        test = f"{100 * complete['final_test_accuracy']:.2f}%" if complete else '—'
        lines.append(f"| {label} | {len(curve)} | {100 * curve[-1]['valAcc']:.2f}% | {test} | "
                     f"{curve[-1]['paramCount']:,} |")
    return lines + ['']


LONG_RUN_GENERATIONS = 15


LONG_RUN_ARMS = (('17a: Monte Carlo 2 search', 'full_growth_long', '#1baa7a'),
                 ('17b: random growth', 'full_growth_long_random', '#eb6834'),
                 ('17c: no growth', 'full_growth_long_static', '#7a7a7a'))


def _selected_actions(directory, searches):
    """Selected action types of a growth run: from its searches, or from its history for random growth."""
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    if searches:
        return [group_key({'action': search['actionChosen']}) for search in searches]
    history = directory / 'history.pt'
    if not history.exists():
        return []
    import torch
    names = torch.load(history, weights_only=False).get('action_name', [])
    return [name.replace('Layer', '').replace('Action', '') for name in names]


def long_run_section():
    """Phase 17: a long overnight growth run from the residual-block starter, against random and no growth."""
    lines = ['', '## Phase 17: does the search beat random growth on a long run?', '',
             'Phase 16c showed that the residual block in the starter is worth about 5 pp. Phase 17 grows the same '
             f'small starter for {LONG_RUN_GENERATIONS} generations of 20 epochs (300 epochs, as long as the '
             'residual-block experiments on main), with three arms on the same schedule and growth candidates '
             '(no dropout, deleting layers or neurons, or shrinking blocks):', '',
             '- **17a**: Monte Carlo 2 with more search: 10 scoring epochs on the CCS 10% simulation set (the best '
             'setting in Phase 15a), 60 s per candidate (5-25 minutes per search), no search started after 7.5 '
             'hours. Seed 1.',
             '- **17b**: the same candidates, one picked at random (a random action type, then a random action of '
             'that type). Seeds 1 and 2.',
             '- **17c**: the starter trained for 300 epochs without growing. Seeds 1 and 2.', '',
             'No experiment so far compared the search with random choice. If 17a does not beat 17b by more than '
             'the seed-to-seed spread (about 3 pp in Phase 12), the search does not pay for its time; 17c shows '
             'how much growth adds at all.', '']
    runs = {}
    for label, folder, color in LONG_RUN_ARMS:
        for seed_dir in sorted((RUN / folder).glob('seed_*')):
            if (seed_dir / 'board/metrics/training.json').exists():
                complete = (seed_dir / 'run.json').exists()
                loaded = _load_growth_run(seed_dir) if complete else _partial_growth_run(seed_dir)
                runs[(label, seed_dir.name)] = (seed_dir, *loaded, color)
    if not runs:
        return lines + ['Pending. Run with `--phase long-run --seeds 1` and `--phase long-run-baselines --seeds 1 2`.',
                        '']
    lines += ['| Arm | Seed | Epochs | Validation at the last epoch | Test | Parameters | Selected actions |',
              '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for (label, seed), (directory, metrics, searches, run, _) in runs.items():
        actions = _selected_actions(directory, searches)
        summary = ', '.join(f'{name} {actions.count(name)}' for name in dict.fromkeys(actions)) or '—'
        test = f"{100 * run['final_test_accuracy']:.2f}%" if run else 'running'
        lines.append(f"| {label} | {seed.split('_')[1]} | {len(metrics)} of {20 * LONG_RUN_GENERATIONS} | "
                     f"{100 * metrics[-1]['valAcc']:.2f}% | {test} | {metrics[-1]['paramCount']:,} | {summary} |")
    fig, (acc, size) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, layout='constrained',
                                    gridspec_kw={'height_ratios': [2, 1.2]})
    for (label, seed), (_, metrics, _, _, color) in runs.items():
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        style = '-' if seed == 'seed_1' else ':'
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], color=color, lw=1.6, ls=style,
                 label=f"{label}, {seed.replace('_', ' ')} (last {100 * metrics[-1]['valAcc']:.1f}%)")
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', color=color, lw=1.6, ls=style)
    for boundary in range(20, 20 * LONG_RUN_GENERATIONS, 20):
        for axis in (acc, size):
            axis.axvline(boundary + .5, color='#dddddd', ls=':', lw=.8)
    acc.set(ylabel='Validation accuracy (%)', ylim=(20, 85),
            title='Phase 17: search, random growth and no growth from the residual-block starter')
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    for axis in (acc, size):
        axis.grid(alpha=.25)
    acc.legend(loc='lower right', fontsize=8)
    fig.savefig(ASSET_DIR / '018-long-run-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    lines += ['', '![Phase 17 training history](/assets/experiments/018-long-run-training.png)', '',
              'Solid lines are seed 1, dotted lines seed 2.', '']
    search_run = runs.get(('17a: Monte Carlo 2 search', 'seed_1'))
    if search_run and search_run[2]:
        _, metrics, searches, _, _ = search_run
        ends = {row['generation']: row for row in metrics if row['epochInGeneration'] == 19}
        lines += ['### 17a decisions', '',
                  '| Search after generation | Selected | Candidates | Mean measurements per candidate | Search time '
                  '| Parameters after | Validation at the end of the next generation |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: |']
        for search in searches:
            generation = search['generation']
            measured = [row for row in search['candidates'] if row.get('visits')]
            visits = sum(row['visits'] for row in measured) / len(measured) if measured else 0
            after = ends.get(generation + 1)
            tail = f"{after['paramCount']:,} | {100 * after['valAcc']:.1f}% |" if after else '— | — |'
            lines.append(f"| {generation} | {_action_label(search['actionChosen'])} | {len(search['candidates'])} | "
                         f"{visits:.1f} | {search['durationSec'] / 60:.0f} min | {tail}")
        lines.append('')
    return lines


COMPARISON_EPOCHS = 30
COMPARISON_ARMS = (('Monte Carlo 1', 'mc_comparison_mc1', '#3b6fd4'),
                   ('Monte Carlo 2', 'mc_comparison_mc2', '#1baa7a'))


def comparison_section():
    """Phase 18: Monte Carlo 1 against Monte Carlo 2 with a search after every generation."""
    total = COMPARISON_EPOCHS * LONG_RUN_GENERATIONS
    shown = COMPARISON_REPORT_EPOCHS
    lines = ['', '## Phase 18: Monte Carlo 1 against Monte Carlo 2', '',
             f'Phase 17 stopped searching after 7.5 hours, so its search grew the network in only 9 of 14 '
             f'generations. Phase 18 drops that deadline and compares the two search algorithms on the Phase 17 '
             f'settings: the residual-block starter, {LONG_RUN_GENERATIONS} generations of {COMPARISON_EPOCHS} '
             f'epochs ({total} epochs), 10 scoring epochs on the CCS 10% simulation set, 60 s per candidate within '
             '5-25 minutes per search, and growth actions only. Seeds 1 and 2 for each algorithm.', '',
             'Both algorithms measure every candidate at least once before they stop, so a search can run past its '
             '25 minutes when there are many candidates.', '']
    runs = {}
    for label, folder, color in COMPARISON_ARMS:
        for seed_dir in sorted((RUN / folder).glob('seed_*')):
            if (seed_dir / 'board/metrics/training.json').exists():
                complete = (seed_dir / 'run.json').exists()
                loaded = _load_growth_run(seed_dir) if complete else _partial_growth_run(seed_dir)
                runs[(label, seed_dir.name)] = (*loaded, color)
    if not runs:
        return lines + ['Pending. Run with `--phase mc1 --seeds 1 2` and `--phase mc2 --seeds 1 2`.', '']
    lines += ['| Algorithm | Seed | Epochs | Training accuracy | Validation accuracy | Test | Parameters | Searches '
              '| Mean search time | Selected actions |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    for (label, seed), (metrics, searches, run, _) in runs.items():
        actions = [group_key({'action': search['actionChosen']}) for search in searches]
        summary = ', '.join(f'{name} {actions.count(name)}' for name in dict.fromkeys(actions)) or '—'
        test = f"{100 * run['final_test_accuracy']:.2f}%" if run else 'running'
        minutes = sum(search['durationSec'] for search in searches) / 60 / len(searches) if searches else 0
        last = metrics[-1]
        lines.append(f"| {label} | {seed.split('_')[1]} | {len(metrics)} of {total} | "
                     f"{100 * last['trainAcc']:.2f}% | {100 * last['valAcc']:.2f}% | {test} | "
                     f"{last['paramCount']:,} | {len(searches)} | {minutes:.0f} min | {summary} |")
    runs = {key: ([row for row in value[0] if row['globalEpoch'] < shown], *value[1:]) for key, value in runs.items()}
    fig, (train, acc, size) = plt.subplots(3, 1, figsize=(11, 15), sharex=True, layout='constrained',
                                           gridspec_kw={'height_ratios': [2, 2, 1.2]})
    for (label, seed), (metrics, _, _, color) in runs.items():
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        style = '-' if seed == 'seed_1' else ':'
        train.plot(epochs, [100 * row['trainAcc'] for row in metrics], color=color, lw=1.6, ls=style,
                   label=f"{label}, {seed.replace('_', ' ')} (last {100 * metrics[-1]['trainAcc']:.1f}%)")
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], color=color, lw=1.6, ls=style,
                 label=f"{label}, {seed.replace('_', ' ')} (last {100 * metrics[-1]['valAcc']:.1f}%)")
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', color=color, lw=1.6, ls=style)
    for axis in (train, acc, size):
        axis.set_xlim(0, shown)
        axis.xaxis.set_major_locator(MultipleLocator(COMPARISON_EPOCHS))
        for boundary in range(COMPARISON_EPOCHS, shown, COMPARISON_EPOCHS):
            axis.axvline(boundary + .5, color='gray', alpha=.4, lw=.8, zorder=0)
        axis.grid(alpha=.25, axis='y')
    top = train.secondary_xaxis('top')
    top.set_xticks([COMPARISON_EPOCHS * (g + .5) for g in range(shown // COMPARISON_EPOCHS)],
                   [str(g) for g in range(shown // COMPARISON_EPOCHS)])
    top.tick_params(length=0, labelsize=7)
    top.set_xlabel('Generation (30 epochs each)')
    train.set(ylabel='Training accuracy (%)', ylim=(20, 95),
              title='Phase 18: Monte Carlo 1 and Monte Carlo 2 from the residual-block starter')
    acc.set(ylabel='Validation accuracy (%)', ylim=(20, 90))
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)')
    size.set_ylim(bottom=0)
    train.legend(loc='lower right', fontsize=8)
    acc.legend(loc='lower right', fontsize=8)
    fig.savefig(ASSET_DIR / '018-mc-comparison-training.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return lines + ['', '![Phase 18 training history](/assets/experiments/018-mc-comparison-training.png)', '',
                    '> [!CAPTION] Figure 31. Phase 18: training accuracy (top), validation accuracy (middle) and '
                    'parameters in thousands (bottom) per epoch, first 300 epochs; solid lines are seed 1, dotted lines seed 2. Gray '
                    'lines mark the generations (30 epochs). Monte Carlo 1 seed 1 and Monte Carlo 2 seed 2 crashed '
                    'at epoch 150; the other two runs are still running.', '',
                    'For the best run, Monte Carlo 2 seed 1, the chart below groups its selected actions by type and '
                    'shows how training and validation accuracy changed over the generation after each one.', '',
                    '![Accuracy change by action type, Monte Carlo 2 seed 1](/assets/experiments/013-mc2-action-types.png)', '',
                    '> [!CAPTION] Figure 99. Phase 18, Monte Carlo 2 seed 1: mean training and validation change over '
                    'the next generation by action type (bars), and every single action (dots). The number of times '
                    'each type was selected is in brackets. Drawn by generate_experiment_013_phase18_details.py.', '',
                    'The network of this run at four points: the starter, two generations in between, and generation 9, '
                    'the last one in the charts. These are the drawings the run saved after each generation.', '',
                    '![Monte Carlo 2 seed 1, starting network](/assets/experiments/013-mc2-structure-gen0.png)', '',
                    '> [!CAPTION] Figure 99. Generation 0, the starter: 904 parameters. An 8-channel convolution with '
                    'BatchNorm, one residual block, pooling and a linear classifier.', '',
                    '![Monte Carlo 2 seed 1, network in generation 3](/assets/experiments/013-mc2-structure-gen3.png)', '',
                    '> [!CAPTION] Figure 99. Generation 3: 19,840 parameters. A convolution with BatchNorm has been added '
                    'after the residual block, and all convolutions are widened from 8 to 32 channels.', '',
                    '![Monte Carlo 2 seed 1, network in generation 6](/assets/experiments/013-mc2-structure-gen6.png)', '',
                    '> [!CAPTION] Figure 99. Generation 6: 189,632 parameters. The stem and block are widened to 64 '
                    'channels and the added convolution to 256.', '',
                    '![Monte Carlo 2 seed 1, network in generation 9](/assets/experiments/013-mc2-structure-gen9.png)', '',
                    '> [!CAPTION] Figure 99. Generation 9: 485,504 parameters. Two more convolutions with BatchNorm were '
                    'inserted inside the residual block branch (64 -> 256 -> 64 channels), so the block itself became '
                    'deeper.', '']


def scoring_calibration_section():
    path = RUN / 'scoring_calibration/gen_3/summary.json'
    lines = ['', '## Phase 15a: which scoring setting predicts real training', '',
             'Phase 13 found no relation between the simulation ranking and real training. Phase 15a reuses '
             "Phase 13's generation-3 model and its 16 candidates, whose real 20-epoch validation accuracy is "
             'known, and scores each of them in five settings: 5, 10 or 15 scoring epochs on a CCS simulation '
             'set of 10% or 25% of the training data. Every arm is measured 3 times with the same three training '
             'seeds, and the score is the mean of the last 2 scoring epochs, as in the search. The learning rate '
             'follows the restarted main schedule (`de231f4`). The best setting becomes the scoring of the next '
             'full-growth run.', '']
    if not path.exists():
        return lines + ['Pending. Run with `--phase scoring-calibration`.', '']
    summary = json.loads(path.read_text(encoding='utf-8'))
    control = summary['control_validation_accuracy']
    lines += ['| Scoring epochs | Simulation set | Rank correlation, 1 measurement | Rank correlation, mean of 3 '
              '| Choice: real rank | Choice: gain over control | Real best: score rank | Real top 5 in score top 5 '
              '| Measurement range | Seconds per measurement |',
              '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary['settings']:
        metrics = row['metrics']
        lines.append(f"| {row['scoring_epochs']} | {row['simulation_set_percent']:g}% | "
                     f"{metrics['spearman_one_measurement']:.2f} | {metrics['spearman_mean']:.2f} | "
                     f"{metrics['chosen_real_rank']} | {100 * metrics['chosen_gain_over_control']:+.2f} pp | "
                     f"{metrics['real_best_rank_by_score']} | {metrics['real_top_in_score_top']} of 5 | "
                     f"{100 * metrics['mean_measurement_range']:.1f} pp | {metrics['seconds_per_measurement']:.0f} |")
    best = max(summary['settings'], key=lambda row: row['metrics']['spearman_mean'])
    return lines + ['', f"The control reached {100 * control:.2f}%. A rank correlation of 1 is the real order, 0 no "
                    'relation. The measurement range is the mean difference between the highest and lowest of the '
                    'three scores of one arm.', '',
                    f"The best agreement is {best['scoring_epochs']} scoring epochs on {best['simulation_set_percent']:g}% "
                    f"of the training data (rank correlation {best['metrics']['spearman_mean']:.2f} with 3 "
                    'measurements).', '']


def sim_fixes_check_section():
    path = RUN / 'sim_fixes_check/results.json'
    lines = ['', '## Phase 11: the A08 decision with the first three simulation fixes', '',
             'The Phase 9 replay repeated with the code at `5d3c3f0`: lazy expansion, the restarted scoring '
             'learning rate and futures that continue from trained models. Everything else matches Phase 9: '
             'the origin run generation-3 checkpoint, the five recorded root actions, 5 scoring epochs with the '
             'mean of the last two, rollout depth 0, UCB C = 1.0, a 5-minute budget and seeds 1–3. This was a quick '
             'check run with a temporary script (saved with the results as `check_script.py`), not an experiment '
             'phase; measuring again before expanding (`abe0a33`) was not yet in the code.', '']
    if not path.exists():
        return lines + ['Pending.', '']
    runs = json.loads(path.read_text(encoding='utf-8'))
    phase6 = json.loads((RUN / 'score_fix_check/summary.json').read_text(encoding='utf-8'))['runs']
    phase9 = json.loads((RUN / 'noise_fix_check/summary.json').read_text(encoding='utf-8'))['runs']
    leads6, leads9 = _a08_lead_values(phase6), _a08_lead_values(phase9)
    a08 = LABELS['candidate_5']
    leads11 = []
    lines += ['| Seed | Selected | A08 score | Next best | A08 lead | Phase 9 lead | Phase 6 lead | Rollouts | Tree depth |',
              '| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |']
    for run, lead6, lead9 in zip(runs, leads6, leads9):
        rows = {row['name']: row for row in run['rows']}
        best_a08 = rows[a08]['score']
        runner_up = max((row for name, row in rows.items() if name != a08), key=lambda row: row['score'])
        lead = 100 * (best_a08 - runner_up['score'])
        leads11.append(lead)
        lines.append(f"| {run['seed']} | {run['selected']} | {best_a08:.4f} | {runner_up['name']} "
                     f"({runner_up['score']:.4f}) | **{lead:+.1f} pp** | {lead9:+.1f} pp | {lead6:+.1f} pp | "
                     f"{run['rollouts']} | {run['depth']} |")
    mean = lambda values: sum(values) / len(values)
    names = [row['name'] for row in runs[0]['rows']]
    lines += ['', 'Each action\'s mean search value, own score and best explored future, averaged over the three '
              'seeds:', '', '| Action | Mean search value | Own score | Best explored future |',
              '| --- | ---: | ---: | ---: |']
    for name in sorted(names, key=lambda n: -mean([next(r['score'] for r in run['rows'] if r['name'] == n)
                                                    for run in runs])):
        rows = [next(r for r in run['rows'] if r['name'] == name) for run in runs]
        futures = [row['futures'][0][1] for row in rows if row['futures']]
        lines.append(f"| {name} | {mean([row['score'] for row in rows]):.4f} | "
                     f"{mean([row['own_score'] for row in rows]):.4f} | "
                     f"{mean(futures):.4f} |" if futures else f"| {name} | — | — | — |")
    widen_own = mean([row['own_score'] for run in runs for row in run['rows'] if row['name'].startswith('Widen')])
    widen_future = mean([row['futures'][0][1] for run in runs for row in run['rows']
                         if row['name'].startswith('Widen') and row['futures']])
    return lines + ['', '### What Phase 11 shows', '',
                    f'- **The gap between the good action and the rest is now bigger.** A08 was selected in all '
                    f'three seeds, and its lead over the next-best action grew from {mean(leads9):+.1f} pp on '
                    f'average in Phase 9 (smallest {min(leads9):+.1f} pp) to **{mean(leads11):+.1f} pp** '
                    f'(smallest {min(leads11):+.1f} pp). Phase 6 had {mean(leads6):+.1f} pp with a tie in one seed.',
                    f'- **Futures now show what widening makes possible.** The two widening actions scored '
                    f'{widen_own:.3f} on their own, but their best follow-up (mostly a convolution with BatchNorm '
                    f'and ReLU on the widened layer) reached {widen_future:.3f}. Before the fix this follow-up was '
                    'built from untrained weights and could not show it.',
                    '- **Searches made fewer rollouts** in the same 5 minutes (19–24 against 26–30 in Phase 9; the '
                    'cause was not measured) and reached tree depth 2–3.',
                    '- **What this does not show:** this decision is at generation 3, where the old scoring learning '
                    'rate was still 0.066, so the learning-rate fix matters little here. Its effect, and the effect of '
                    'measuring again, appear in late generations of a full-growth run. Deeper futures also have more '
                    'scoring training in total (10 epochs at depth 2 against 5 at the root), so part of the higher '
                    'future scores is extra training.', '']


def simulation_fixes_after_phase10_section():
    return ['', '## Simulation fixes after Phase 10', '',
            'The Phase 10 analysis traced the near-flat candidate scores to three problems in how Monte Carlo 2 '
            'built and trained candidates. All three are fixed in the code; Phase 11 checks them on the A08 '
            'decision. A fourth fix, measuring again before expanding, was added after Phase 11; Phase 12 runs '
            'full growth with all four.', '',
            '| Fix | Problem found in Phase 10 | Change | Commit |',
            '| --- | --- | --- | --- |',
            '| **Lazy expansion** | Expanding a node built every child (about 70 model copies with every action '
            'applied) although the search scores one per step. Rollouts took about 20 s instead of 10–12 s, so the '
            'budget bought about 1.2 measurements per candidate. | A child model is built only when the search first '
            'visits it. The search rules and the order of children are unchanged. | `f8bed63` |',
            '| **Scoring learning rate** | Scoring trained each candidate with the main run\'s learning rate at its '
            'current point: 0.098 in generation 0 but 0.010 and 0.0032 in generations 7–8. Late candidates barely '
            'changed during scoring, so up to 34 of 77 scored within 1 pp of the best. | Every scoring run trains '
            'with a restarted copy of the main schedule (`lr_scheduler.restarted()`), so candidates in every '
            'generation are trained the way the main run starts. The main run\'s schedule is never modified. | '
            '`de231f4` |',
            '| **Futures continue from trained models** | Scoring trained a throwaway copy, so a child "X then Y" '
            'was built from the untrained parent and evaluated as X and Y applied at once, never as X → train → Y '
            '→ train. An action that pays off one step later, such as widening, could not show its value. | '
            '`SimulationScore.score_and_train` also returns the copy its first training term trained; each node '
            'keeps it, and its children and rollout follow-ups are built from it. | `5d3c3f0` |',
            '| **Measure again before expanding** *(added after Phase 11)* | Each node\'s own score was measured '
            'once, and the best of 40–77 single noisy scores tends to be a lucky one rather than the best action. | '
            'A node is measured `MCTS_MEASUREMENTS_BEFORE_EXPAND` times (default 2), measurement *i* with training '
            'seed *search seed + i*, before it is expanded; its value is the average. A re-measurement replaces an '
            'expansion visit, so a search makes the same number of scoring runs. | `abe0a33` |',
            '', 'Not adopted from the Phase 10 suggestions: rules that prefer adding capacity when scores are close '
            '(suggestions 1–2 above). The search must pick by its score alone, so that the same algorithm still '
            'works when the score rewards fewer parameters. The learning-rate schedule of the main run (suggestion '
            '3) is a training-recipe question and is left for a separate experiment.', '',
            'Expected effect, to be checked in the next run: late-generation candidates should separate instead of '
            'tying within 1 pp, and each search should measure more candidates more than once. One side effect to '
            'watch: a depth-2 candidate has had 10 scoring epochs in total against 5 for a root action, so deeper '
            'measurements may score higher.', '']


def full_grid_section():
    lines = ['', '## Phase 10: UCB C and simulation-set grid (full growth)', '',
             'Full growth from the Experiment 010 starter, as in Phase 7, with the Phase 9 search: fixed '
             'Monte Carlo 2, type-balanced expansion, one training seed per search, the mean of the last two '
             'of 5 scoring epochs, rollout depth 0, and convolution widening including fan-out widening. '
             'The search budget grows with the number of candidates: 24 seconds per candidate (about two '
             'rollouts), at least 5 minutes.', '',
             'Two factors are crossed:', '',
             '- **UCB C = 0.05 or 1.0.** Score gaps between good actions are 0.01–0.05. With C = 1.0 the '
             'exploration bonus is about ten times larger, so visits spread evenly (Phase 9). With C = 0.05, '
             'extra visits concentrate on the leading candidates.',
             '- **Simulation set = Moderate or CCS** (Experiment 011). Moderate was best and most stable on '
             'CIFAR-10 (58.4%, seeds 57.9–58.8%); CCS was best over CIFAR-10 and MNIST together (75.2%), '
             'mostly from MNIST, and varied strongly between CIFAR-10 seeds (46.1–61.4%).', '',
             'Seed 1 is run for all four cells first. Each factor is judged on two runs against two; seed 2 '
             'repeats the grid if time allows.', '',
             '| UCB C | Simulation set | Seed 1 validation | Seed 2 validation | Mean validation | Mean test | Final parameters | Search time per run |',
             '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    runs = _grid_runs()
    for ucb_c, selector in GRID_CELLS:
        seeds = runs.get((ucb_c, selector), {})
        cells = [f"{100 * seeds[seed]['final_validation_accuracy']:.2f}%" if seed in seeds else 'Pending'
                 for seed in (1, 2)]
        if seeds:
            done = list(seeds.values())
            mean = lambda key: sum(row[key] for row in done) / len(done)
            summary = (f"{100 * mean('final_validation_accuracy'):.2f}% | {100 * mean('final_test_accuracy'):.2f}% | "
                       f"{mean('final_parameters'):,.0f} | {mean('search_minutes'):.0f} min")
        else:
            summary = '— | — | — | —'
        lines.append(f'| {ucb_c} | {GRID_SET_NAMES[selector]} | ' + ' | '.join(cells) + f' | {summary} |')
    complete = [seed for seed in (1, 2) if all(seed in runs.get(cell, {}) for cell in GRID_CELLS)]
    if complete:
        def factor_mean(predicate):
            values = [runs[cell][seed]['final_validation_accuracy'] for cell in GRID_CELLS if predicate(cell)
                      for seed in complete]
            return 100 * sum(values) / len(values)
        lines += ['', f"Average final validation over the complete seeds ({', '.join(map(str, complete))}):", '',
                  '| Factor | Level | Mean validation |', '| --- | --- | ---: |',
                  f'| UCB C | 0.05 | {factor_mean(lambda cell: cell[0] == 0.05):.2f}% |',
                  f'| UCB C | 1.0 | {factor_mean(lambda cell: cell[0] == 1.0):.2f}% |',
                  f'| Simulation set | Moderate | {factor_mean(lambda cell: cell[1] == "moderate"):.2f}% |',
                  f'| Simulation set | CCS | {factor_mean(lambda cell: cell[1] == "ccs"):.2f}% |']
        lines += _grid_figures(runs, complete)
        lines += _grid_decisions(1) + _grid_findings(runs, complete)
    else:
        lines += ['', 'Run with `--phase full-grid --seeds 1` (all four cells), then regenerate this report.']
    return lines + ['']


GRID_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']  # validated categorical slots 1-4, in cell order
ASSET_DIR = ROOT / 'documentation/website/app/public/assets/experiments'


def _cell_name(ucb_c, selector):
    return f"c{str(ucb_c).replace('.', 'p')}_{selector}"


def _cell_label(ucb_c, selector):
    return f'C={ucb_c}, {GRID_SET_NAMES[selector]}'


def _grid_metrics(ucb_c, selector, seed):
    board = RUN / 'full_grid' / _cell_name(ucb_c, selector) / f'seed_{seed}' / 'board'
    return json.loads((board / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']


CAPACITY_NEUTRAL = {'Dropout', 'Add neurons', 'Delete neurons', 'Delete layer'}
EXP13_RUN = ROOT / 'experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1'


def _training_comparison(cells, title, filename, seed):
    """Training (dashed) and validation (solid) accuracy plus parameters for two cells."""
    fig, (acc, size) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    for (ucb_c, selector), color in cells:
        metrics = _grid_metrics(ucb_c, selector, seed)
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        label = _cell_label(ucb_c, selector)
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], color=color, lw=1.6,
                 label=f"{label} validation (final {100 * metrics[-1]['valAcc']:.1f}%)")
        acc.plot(epochs, [100 * row['trainAcc'] for row in metrics], color=color, lw=1.2, ls='--',
                 label=f"{label} training (final {100 * metrics[-1]['trainAcc']:.1f}%)")
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', color=color, lw=1.6, label=label)
    acc.set(ylabel='Accuracy (%)', ylim=(0, 80), title=title)
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    for axis in (acc, size):
        axis.grid(alpha=.25)
        for boundary in range(20, 200, 20):
            axis.axvline(boundary + .5, color='#bbbbbb', ls=':', lw=.8)
    acc.legend(loc='lower right', fontsize=8)
    size.legend(loc='upper left', fontsize=8)
    fig.savefig(ASSET_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _grid_figures(runs, complete):
    seed = complete[0]
    best = max(GRID_CELLS, key=lambda cell: runs[cell][seed]['final_validation_accuracy'])
    other_c = next(cell for cell in GRID_CELLS if cell[1] == best[1] and cell[0] != best[0])
    other_set = next(cell for cell in GRID_CELLS if cell[0] == best[0] and cell[1] != best[1])
    color = dict(zip(GRID_CELLS, GRID_COLORS))
    _training_comparison([(other_c, color[other_c]), (best, color[best])],
                         f'UCB C compared ({GRID_SET_NAMES[best[1]]} simulation set), seed {seed}',
                         '018-grid-training-ucb.png', seed)
    _training_comparison([(other_set, color[other_set]), (best, color[best])],
                         f'Simulation set compared (C = {best[0]}), seed {seed}',
                         '018-grid-training-simset.png', seed)

    # Factor means: one panel per factor, bars from zero, individual runs as dots.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True, layout='constrained')
    factors = [('UCB C', [(0.05, '0.05'), (1.0, '1.0')], 0), ('Simulation set', [('moderate', 'Moderate'), ('ccs', 'CCS')], 1)]
    for ax, (title, levels, index) in zip(axes, factors):
        for position, (level, name) in enumerate(levels):
            values = [100 * runs[cell][s]['final_validation_accuracy'] for cell in GRID_CELLS
                      if cell[index] == level for s in complete]
            mean = sum(values) / len(values)
            ax.bar(position, mean, width=.55, color='#2a78d6')
            ax.scatter([position] * len(values), values, s=36, color='#3d3d3a', zorder=3)
            ax.text(position, max([mean, *values]) + 1.5, f'{mean:.2f}%', ha='center', fontsize=10, color='#1f1f1d')
        ax.set_xticks(range(len(levels)), [name for _, name in levels])
        ax.set(title=title, ylim=(0, 75))
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Final validation accuracy (%)')
    fig.suptitle(f'Phase 10: mean final validation per factor level (dots: individual runs, seed {seed})')
    fig.savefig(ASSET_DIR / '018-grid-factor-means.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # How well each search separated its candidates: every candidate's score minus the best.
    fig, axes = plt.subplots(1, len(GRID_CELLS), figsize=(16, 4.6), sharey=True, layout='constrained')
    for ax, cell in zip(axes, GRID_CELLS):
        for search in _grid_searches(*cell, seed):
            scores = [row['score'] for row in search['candidates']]
            top = max(scores)
            gaps = [100 * (score - top) for score in scores]
            jitter = [search['generation'] + .5 * ((i * 0.618) % 1 - .5) for i in range(len(gaps))]
            ax.scatter(jitter, gaps, s=9, color=color[cell], alpha=.55, linewidths=0)
            ties = sum(gap > -1 for gap in gaps)
            ax.text(search['generation'], 1.2, f'{ties}/{len(gaps)}', ha='center', fontsize=7, color='#3d3d3a')
        ax.axhspan(-1, 0, color='#cccccc', alpha=.45, lw=0)
        ax.set(title=_cell_label(*cell), xlabel='Generation', xticks=range(9), ylim=(-20, 3))
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('Score minus best candidate (pp)')
    fig.suptitle(f'How clearly each search separated its candidates, seed {seed} '
                 '(grey band: within 1 pp of the best; numbers: candidates inside it)')
    fig.savefig(ASSET_DIR / '018-grid-separation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # What each selected action delivered, against the origin run.
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    styles = {'adds layer': ('o', '#2a78d6'), 'widens': ('D', '#1baf7a'), 'no capacity': ('x', '#eb6834')}
    plotted = set()
    for index, cell in enumerate(GRID_CELLS):
        metrics = _grid_metrics(*cell, seed)
        end = {row['globalEpoch'] // 20: row['valAcc'] for row in metrics}
        for search in _grid_searches(*cell, seed):
            generation = search['generation']
            kind = group_key({'action': search['actionChosen']})
            category = ('widens' if kind == 'Widen convolution' else
                        'no capacity' if kind in CAPACITY_NEUTRAL else 'adds layer')
            marker, mark_color = styles[category]
            ax.scatter(generation + (index - 1.5) * .12, 100 * (end[generation + 1] - end[generation]),
                       marker=marker, color=mark_color, s=40,
                       label={'adds layer': 'Phase 10: adds a layer', 'widens': 'Phase 10: widens a convolution',
                              'no capacity': 'Phase 10: adds no capacity'}[category] if category not in plotted else None)
            plotted.add(category)
    exp13 = json.loads((EXP13_RUN / 'board/metrics/training.json').read_text(encoding='utf-8'))['epochs']
    end13 = {row['globalEpoch'] // 20: row['valAcc'] for row in exp13}
    ax.plot(range(9), [100 * (end13[g + 1] - end13[g]) for g in range(9)], color='#6b6b66', lw=1.4,
            marker='.', label='the origin run (widened to 64 channels, then A08 at generation 3)')
    ax.axhline(0, color='#999999', lw=.8)
    ax.set(xlabel='Generation in which the action was chosen', xticks=range(9),
           ylabel='Validation change over the next generation (pp)',
           title=f'What each selected action delivered (all four Phase 10 cells, seed {seed})')
    ax.grid(axis='y', alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(ASSET_DIR / '018-grid-action-gains.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Detailed candidate chart for the best cell only.
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import COLORS, GROUPS
    searches = _grid_searches(*best, seed)
    groups = [(key, name.replace('\n', ' ')) for key, name in GROUPS
              if any(group_key(row) == key for search in searches for row in search['candidates'])]
    fig, axes = plt.subplots(len(searches), 1, figsize=(15, 2.6 * len(searches)), squeeze=False,
                             sharex=True, sharey=True)
    fig.suptitle(f'Phase 10 {_cell_label(*best)}, seed {seed}: candidate search scores by action type',
                 fontsize=14, weight='bold')
    for ax, search in zip(axes.flat, searches):
        for position, (key, _) in enumerate(groups):
            rows = [row for row in search['candidates'] if group_key(row) == key and row.get('score') is not None]
            for j, row in enumerate(rows):
                offset = .56 * (j - (len(rows) - 1) / 2) / max(len(rows) - 1, 1)
                chosen = row['action'] == search['actionChosen']
                mark = (COLORS['Sequential convolution'] if key.startswith('Sequential convolution')
                        else COLORS.get(key, COLORS['Other']))
                ax.scatter(position + offset, row['score'], s=120 if chosen else 22,
                           marker='*' if chosen else 'o', color=mark,
                           edgecolors='black' if chosen else 'none', zorder=3)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Gen {search['generation']}\nscore")
        ax.grid(axis='y', alpha=.25)
    axes[-1, 0].set_xticks(range(len(groups)), [label for _, label in groups], rotation=25, ha='right')
    fig.subplots_adjust(top=.965, bottom=.04, hspace=.22)
    best_name = f'018-grid-{_cell_name(*best)}-scores.png'
    fig.savefig(ASSET_DIR / best_name, dpi=120, bbox_inches='tight')
    plt.close(fig)

    return ['', '### Training histories', '',
            f'Solid lines are validation accuracy, dashed lines training accuracy. The best cell '
            f'({_cell_label(*best)}) is compared with the other UCB C at the same simulation set, and with the '
            'other simulation set at the same C.', '',
            '![UCB C compared](/assets/experiments/018-grid-training-ucb.png)', '',
            '![Simulation set compared](/assets/experiments/018-grid-training-simset.png)', '',
            '### Factor means', '',
            '![Phase 10 mean final validation per factor level](/assets/experiments/018-grid-factor-means.png)', '',
            'Each bar averages the two runs at that level (one per level of the other factor); dots are the '
            'individual runs.', '',
            '### Could the search tell its candidates apart?', '',
            '![How clearly each search separated its candidates](/assets/experiments/018-grid-separation.png)', '',
            'Each dot is one candidate, placed by how far its score was below the best candidate of that search. '
            'The grey band is the 1 pp noise level of one scoring run; the number above each generation counts the '
            'candidates inside it. A search that knows what is best has one dot at 0 and the rest well below; many '
            'dots inside the band mean the choice was effectively a coin toss among near-ties.', '',
            '### What the selected actions delivered', '',
            '![What each selected action delivered](/assets/experiments/018-grid-action-gains.png)', '',
            'Validation change over the generation after each decision, for every decision in the four cells, '
            'by kind of action. The grey line is the origin run for contrast: its generation-3 action (A08, '
            '64 → 64 convolution after widening to 64 channels) gave the +20.8 pp jump.', '',
            '### Candidate scores in the best cell', '',
            f'![Candidate scores, {_cell_label(*best)}](/assets/experiments/{best_name})', '',
            'Each row is one generation and each dot one candidate, grouped by action type; the black-edged '
            'star is the selected action.']


def _grid_findings(runs, complete):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    seed = complete[0]
    searches = [search for cell in GRID_CELLS for search in _grid_searches(*cell, seed)]
    late = [search for search in searches if search['generation'] >= 3]
    candidates = sum(len(search['candidates']) for search in late)
    rollouts = sum(search['rollouts'] for search in late)
    seconds = sum(search['durationSec'] for search in late) / rollouts
    widen_offered = sum(any('Widen Conv' in row['action'] for row in search['candidates']) for search in searches)
    widen_chosen = sum('Widen Conv' in search['actionChosen'] for search in searches)
    values = [runs[cell][s]['final_validation_accuracy'] * 100 for cell in GRID_CELLS for s in complete]

    gains = {'neutral': [], 'adds': [], 'early': []}
    train_end, val_end = [], []
    for cell in GRID_CELLS:
        metrics = _grid_metrics(*cell, seed)
        end = {row['globalEpoch'] // 20: row for row in metrics}
        train_end.append(100 * metrics[-1]['trainAcc'])
        val_end.append(100 * metrics[-1]['valAcc'])
        for search in _grid_searches(*cell, seed):
            g = search['generation']
            gain = 100 * (end[g + 1]['valAcc'] - end[g]['valAcc'])
            if g <= 2:
                gains['early'].append(gain)
            elif group_key({'action': search['actionChosen']}) in CAPACITY_NEUTRAL:
                gains['neutral'].append(gain)
            else:
                gains['adds'].append(gain)
    lr = {row['globalEpoch'] // 20: row['lr'] for row in _grid_metrics(*GRID_CELLS[0], seed)}

    widen_vs_quiet, layer_vs_quiet = [], []
    for search in searches:
        def best(match):
            scores = [row['score'] for row in search['candidates'] if match(row)]
            return max(scores) if scores else None
        quiet = best(lambda row: 'Dropout' in row['action'] and 'p=0.1' in row['action'])
        widen = best(lambda row: 'Widen Conv' in row['action'])
        layer = best(lambda row: group_key(row) in ('Sequential convolution + BN', 'Sequential convolution without BN',
                                                     'Residual convolution'))
        if quiet is not None and widen is not None:
            widen_vs_quiet.append(100 * (widen - quiet))
        if quiet is not None and layer is not None:
            layer_vs_quiet.append(100 * (layer - quiet))
    ties = [sum(max(row['score'] for row in s['candidates']) - row['score'] < 0.01 for row in s['candidates'])
            for s in searches if s['generation'] >= 7]
    late_sizes = [len(s['candidates']) for s in searches if s['generation'] >= 7]
    mean = lambda items: sum(items) / len(items)
    exp13 = json.loads((EXP13_RUN / 'board/metrics/training.json').read_text(encoding='utf-8'))['epochs']
    exp13_params = {row['globalEpoch'] // 20: row['paramCount'] for row in exp13}
    final_params = [runs[cell][s]['final_parameters'] for cell in GRID_CELLS for s in complete]

    return ['', '### What Phase 10 shows', '',
            f'- **Final validation ranged from {min(values):.2f}% to {max(values):.2f}%.** No cell beat Phase 5 '
            '(64.02%) and all stayed far below the origin run (69.48%). C = 1.0 ended higher than C = 0.05 with both '
            'simulation sets; with one seed per cell this is within the seed-to-seed spread. Moderate and CCS were '
            'practically equal.',
            f'- **Nothing useful was added after generation 2.** Generations 0–2 gained {mean(gains["early"]):+.1f} pp '
            f'on average. From generation 3, generations where the selected action added a layer gained '
            f'{mean(gains["adds"]):+.1f} pp, and generations where it added no capacity (dropout, neuron or layer '
            f'removal) gained {mean(gains["neutral"]):+.1f} pp. The added layers brought nothing beyond continued '
            'training.',
            f'- **The models are capacity-limited, not overfitting.** Final training accuracy was '
            f'{min(train_end):.1f}–{max(train_end):.1f}% against {min(val_end):.1f}–{max(val_end):.1f}% validation '
            '(training is measured with cutout augmentation). The networks cannot fit even the training data, so only '
            'a real capacity increase can help.',
            f'- **The network stayed 8 channels wide.** The only widening chosen took conv1 to 16 channels in '
            f'generation 7 of C = 1.0, Moderate, too late to matter. Every added convolution is 8 → 8 '
            f'(about +600 parameters) and everything reaches the classifier through an 8-channel global pool. Final '
            f'models had {min(final_params):,}–{max(final_params):,} parameters. the origin run widened conv1 to 64 '
            f'channels in generations 0–2 and had {exp13_params[4]:,} parameters after its generation-3 action.',
            f'- **The score cannot see the value of widening.** Widening was offered in {widen_offered} of '
            f'{len(searches)} searches and chosen {widen_chosen} {"time" if widen_chosen == 1 else "times"}. Its best '
            f'score was on average {mean(widen_vs_quiet):+.1f} pp from a nearly-no-change action (dropout 0.1), while '
            f'the best added layer scored {mean(layer_vs_quiet):+.1f} pp above it. Widening copies channels so the '
            'function is unchanged, and the copies only drift apart slowly, so within 5 scoring epochs a widened '
            'model behaves like the unwidened one. An added layer gets a small short-term edge that does not last.',
            f'- **The scores flatten as the model trains.** Every candidate starts from the same trained weights and '
            f'changes it only slightly, so in generations 7–8 between {min(ties)} and {max(ties)} of '
            f'{min(late_sizes)}–{max(late_sizes)} candidates scored within 1 pp of the best (the separation chart).',
            f'- **Late changes have little learning rate left.** The cosine schedule falls from 0.1 to '
            f'{lr[5]:.3f} at the end of generation 5 and {lr[8]:.4f} at the end of generation 8, so a new layer added '
            'late cannot learn much in the remaining epochs. the origin run made its decisive change at generation 3, '
            f'with the learning rate still at {lr[3]:.3f}.',
            f'- **The budget bought fewer rollouts than planned:** {rollouts / candidates:.1f} per candidate from '
            f'generation 3, at about {seconds:.0f} s per rollout instead of 10–12 s in Phase 9. Expanding a node builds '
            'every child up front, which is the most likely cause (not profiled). As a result UCB C changed little '
            'after the first generations.', '',
            '### What to change', '',
            'The search is not failing to find a good action among those it compares; it compares the wrong things '
            'with a score that cannot see width. In order of expected impact:', '',
            '1. **Make width growth reachable.** Offer widening by larger factors (up to 64 channels) and favour '
            'capacity when scores are within the 1 pp noise level, for example by picking the near-tied candidate '
            'that adds the most parameters. This targets both the 8-channel bottleneck and the near-ties.',
            '2. **Score the gain over the unchanged model.** Score the current model once per search and rank '
            'candidates by improvement over it; if no candidate beats it by more than the noise, prefer the one that '
            'adds most capacity instead of a random near-tie.',
            '3. **Decide big changes early or keep the learning rate up.** Either limit growth to the first half of '
            'the run, or restart the learning rate after each structural change, so late additions can still learn.',
            '4. **Build children lazily** so the budget buys the intended two or more rollouts per candidate.',
            '', 'A cheap way to confirm the diagnosis before changing the search: one run that widens conv1 by 2× in '
            'each of generations 0–2 (as the origin run did) and then lets the Phase 10 search decide. If it reaches '
            'the origin run territory, width is the missing ingredient and items 1–2 are the right fix.']


def _grid_searches(ucb_c, selector, seed):
    board = RUN / 'full_grid' / f"c{str(ucb_c).replace('.', 'p')}_{selector}" / f'seed_{seed}' / 'board'
    return sorted((json.loads(path.read_text(encoding='utf-8'))
                   for path in (board / 'simulations').glob('simulation_gen_*.json')),
                  key=lambda search: search['generation'])


def _grid_decisions(seed):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    lines = ['', f'### Seed {seed} decisions', '',
             'Selected action type in each generation, with the rank of the best widening candidate '
             '(by mean score) in brackets.', '',
             '| Generation | ' + ' | '.join(f'C={c}, {GRID_SET_NAMES[s]}' for c, s in GRID_CELLS) + ' |',
             '| ---: |' + ' --- |' * len(GRID_CELLS)]
    per_cell = [_grid_searches(c, s, seed) for c, s in GRID_CELLS]
    for generation in range(max(len(searches) for searches in per_cell)):
        cells = []
        for searches in per_cell:
            if generation >= len(searches):
                cells.append('—')
                continue
            search = searches[generation]
            ranked = sorted(search['candidates'], key=lambda row: row['score'], reverse=True)
            widen = [rank for rank, row in enumerate(ranked, 1) if 'Widen Conv' in row['action']]
            cells.append(f"{group_key({'action': search['actionChosen']})} (#{widen[0] if widen else '–'})")
        lines.append(f'| {generation} | ' + ' | '.join(cells) + ' |')
    return lines


def noise_fix_section():
    path = RUN / 'noise_fix_check/summary.json'
    lines = ['', '## Phase 9: replaying the A08 decision with noise fixes and a 5-minute budget', '',
             'Phases 6 and 8 stopped as soon as each root action had one rollout, so they measured a single '
             'noisy score per action and no future actions. Phase 9 uses the same five root actions, 5 scoring '
             'epochs, rollout depth 0 and seeds 1–3, with four changes:', '',
             '- **5-minute budget** (instead of 32 seconds), so the search keeps going after the first pass: '
             'it expands root actions and scores "action + next action" models, which both measures the '
             'future and adds more measurements per root action.',
             '- **Type-balanced expansion order.** New children are tried in a random order that cycles '
             'through action types, instead of the fixed action-list order.',
             '- **Type-balanced random follow-ups** (inactive here, since rollout depth is 0). Rollout follow-ups pick an action type first, so dropout '
             'variants no longer dominate.',
             '- **One training seed per search, and a 2-epoch score.** Every scoring run in a search uses the '
             'same training seed, so differences come from the architectures; the score averages the last '
             'two scoring epochs.', '',
             'Cells show the mean search value and visit count; **bold** marks the selected action.', '',
             *CHECK_TABLE_HEADER]
    if not path.exists():
        lines.append('| — | Pending | Pending | Pending | Pending | Pending | Pending | Pending |')
        return lines + ['', 'Run the experiment script with `--phase check-noise-fix --seeds 1 2 3`, then '
                        'regenerate this report.', '']
    runs = json.loads(path.read_text(encoding='utf-8'))['runs']
    rows, result = _check_table(runs)
    lines += rows + ['', result, '',
                     '| Seed | Rollouts | Tree depth | A08 lead over the next-best action |',
                     '| ---: | ---: | ---: | ---: |']
    for run, lead in zip(runs, _a08_leads(runs).removesuffix(' pp').split(', ')):
        lines.append(f"| {run['seed']} | {run['rollouts']} | {run['max_depth']} | {lead} pp |")
    phase6 = RUN / 'score_fix_check/summary.json'
    if phase6.exists():
        runs6 = json.loads(phase6.read_text(encoding='utf-8'))['runs']
        lines += ['', f'For comparison, Phase 6 (one rollout per action) had A08 leads of {_a08_leads(runs6)}.']
        lines += _noise_fix_findings(runs, runs6)
    return lines + ['']


def _a08_lead_values(runs):
    values = []
    for run in runs:
        scores = {row['arm']: row['score'] for row in run['results']}
        a08 = scores.pop('candidate_5')
        values.append(100 * (a08 - max(scores.values())))
    return values


def _noise_fix_findings(runs, runs6):
    leads, leads6 = _a08_lead_values(runs), _a08_lead_values(runs6)
    a08_wins = sum(lead > 0 for lead in leads)
    visits = [row['visits'] for run in runs for row in run['results']]
    rollouts = [run['rollouts'] for run in runs]
    per_rollout = [run['duration_sec'] / run['rollouts'] for run in runs]
    return ['', '### What Phase 9 shows', '',
            f'- **A08 was selected in {a08_wins} of {len(runs)} seeds**, with a strict lead in each. '
            'This is the first multi-seed replay in which the search picked A08 every time.',
            f'- **The weakest seed improved, the average lead did not change much.** The smallest A08 lead '
            f'rose from {min(leads6):+.1f} pp (Phase 6) to {min(leads):+.1f} pp; the mean lead was '
            f'{sum(leads6) / len(leads6):+.1f} pp in Phase 6 and {sum(leads) / len(leads):+.1f} pp here. '
            'The gain is reliability, not a bigger gap.',
            f'- **Every action got about the same number of visits** ({min(visits)}–{max(visits)} each, '
            f'{min(rollouts)}–{max(rollouts)} rollouts per search). With UCB C=1.0 the exploration bonus '
            'is roughly ten times larger than the score gaps of 0.02–0.06, so UCB spread the visits evenly. '
            'In effect each action was measured about six times, itself plus "action + next action" models, '
            'and the averages were compared.',
            '- **Scores are lower than in Phase 6 by design.** Each average now includes follow-up models '
            'and the mean of the last two scoring epochs, so only the ranking is comparable with earlier '
            'phases, not the absolute values.',
            '- **The four changes were tested together**, so this phase does not show which one mattered. '
            'The longer budget is the most likely main factor, because it is what gives each action several '
            'measurements.',
            f'- **Caveat for full growth.** One rollout took {min(per_rollout):.0f}–{max(per_rollout):.0f} '
            'seconds here, so a 5-minute search manages about 30 rollouts. This replay has only five root '
            'actions; full-growth generations had 18–58, so most would still get a single visit within '
            'five minutes. Getting the same benefit there needs a budget that grows with the number of '
            'candidates, or a smaller UCB C so that visits concentrate on the leaders.']


def score_fix_20_epochs_section():
    path = RUN / 'score_fix_check_20ep/summary.json'
    lines = ['', '## Phase 8: replaying the A08 decision with 20 scoring epochs', '',
             'Phase 6 repeated with **20 scoring epochs** instead of 5; everything else is unchanged '
             '(fixed Monte Carlo 2, rollout depth 0, convolution widening, UCB C=1.0, minimum iterations 3, '
             '32-second soft budget, seeds 1–3). Phase 6 showed that one 5-epoch score per action varies by '
             'several points between seeds, so A08 tied with Widen 128 in seed 2. This phase measures whether '
             'longer scoring makes the scores more stable and A08\'s lead clearer.', '',
             *CHECK_TABLE_HEADER]
    if not path.exists():
        lines.append('| — | Pending | Pending | Pending | Pending | Pending | Pending | Pending |')
        return lines + ['', 'Run the experiment script with `--phase check-score-fix-20 '
                        '--seeds 1 2 3`, then regenerate this report.', '']
    runs = json.loads(path.read_text(encoding='utf-8'))['runs']
    rows, result = _check_table(runs)
    lines += rows + ['', result]
    phase6 = RUN / 'score_fix_check/summary.json'
    if phase6.exists():
        runs5 = json.loads(phase6.read_text(encoding='utf-8'))['runs']
        lines += ['', '### 5 vs 20 scoring epochs', '',
                  'Mean score across seeds, with the lowest and highest seed in brackets. A narrower range '
                  'means a single scoring run is more reliable.', '',
                  '| Action | 5 epochs (Phase 6) | 20 epochs (Phase 8) |', '| --- | ---: | ---: |']
        for i in range(1, 6):
            arm = f'candidate_{i}'
            lines.append(f'| {LABELS[arm]} | {_spread(runs5, arm)} | {_spread(runs, arm)} |')
        lines.append(f'| **A08 lead over the next-best action, per seed** | {_a08_leads(runs5)} | {_a08_leads(runs)} |')
        mean_time = lambda items: sum(run['duration_sec'] for run in items) / len(items) / 60
        lines.append(f'| Search time per seed | {mean_time(runs5):.1f} min | {mean_time(runs):.1f} min |')
        lead5, lead20 = _averaged_a08_lead(runs5), _averaged_a08_lead(runs)
        lines += ['', '### What Phase 8 shows', '',
                  '- **20 scoring epochs did not make a single score reliable.** A08 again lost seed 2, now '
                  'by about one validation image, and the spread of each action across seeds is about as '
                  'wide as with 5 epochs. Seed 2 was unfavourable for A08 in both settings, so the noise '
                  'comes mostly from the random training draw of one scoring run, not from too few epochs.',
                  f'- **Averaging seeds does separate A08.** Averaged over the three seeds, A08 leads the '
                  f'next-best action by {lead5:+.1f} pp with 5 epochs and {lead20:+.1f} pp with 20 epochs.',
                  f'- **Averaging is also cheaper.** Three 5-epoch runs per action take about '
                  f'{3 * mean_time(runs5):.1f} minutes for these five actions, less than one 20-epoch run '
                  f'({mean_time(runs):.1f} minutes).',
                  '- **Implication for the search:** re-scoring the top few candidates with different seeds '
                  'and averaging is a better use of compute than longer scoring. This was not yet tested in '
                  'a full-growth run.']
    return lines + ['']


def _averaged_a08_lead(runs):
    means = {f'candidate_{i}': sum(next(row['score'] for row in run['results'] if row['arm'] == f'candidate_{i}')
                                   for run in runs) / len(runs) for i in range(1, 6)}
    a08 = means.pop('candidate_5')
    return 100 * (a08 - max(means.values()))


PHASE7_CHART = ROOT / 'documentation/website/app/public/assets/experiments/018-phase7-seed1-vs-phase5.png'


def _load_growth_run(directory):
    """Per-epoch metrics and per-generation searches of one full-growth seed."""
    board = directory / 'board'
    metrics = json.loads((board / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']
    searches = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
        (board / 'simulations').glob('simulation_gen_*.json'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))]
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    return metrics, searches, run


def _action_label(action):
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import candidate_detail, group_key
    row = {'action': action}
    detail = ' / '.join(part for part in candidate_detail(row).splitlines() if part)
    return f'{group_key(row)} — {detail}' if detail else group_key(row)


def _phase7_seed1_analysis(target):
    phase5 = _load_growth_run(RUN / 'full_growth_5min' / 'seed_1')
    phase7 = _load_growth_run(target / 'seed_1')
    fig, (acc, size) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    for (metrics, _, _), name, color in ((phase5, 'Phase 5', '#888888'), (phase7, 'Phase 7', '#e56b28')):
        epochs = [row['globalEpoch'] + 1 for row in metrics]
        acc.plot(epochs, [100 * row['valAcc'] for row in metrics], label=f'{name} validation', color=color)
        size.step(epochs, [row['paramCount'] / 1000 for row in metrics], where='post', label=name, color=color)
    acc.set(ylabel='Validation accuracy (%)', ylim=(0, 100), title='Experiment 013 seed 1: Phase 7 vs Phase 5')
    size.set(xlabel='Training epoch', ylabel='Parameters (thousands)', ylim=(0, None))
    for axis in (acc, size):
        axis.grid(alpha=.25)
        axis.legend()
        for boundary in range(20, 200, 20):
            axis.axvline(boundary + .5, color='#999999', ls=':', lw=.8)
    PHASE7_CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PHASE7_CHART, dpi=150, bbox_inches='tight')
    plt.close(fig)

    lines = ['', '### Seed 1 compared with Phase 5', '',
             '| Run | Final validation | Final test | Best validation | Final parameters | Total search time | Longest search |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, (metrics, searches, run) in (('Phase 5', phase5), ('Phase 7', phase7)):
        durations = [search['durationSec'] / 60 for search in searches]
        lines.append(f"| {name} | {100 * run['final_validation_accuracy']:.2f}% | "
                     f"{100 * run['final_test_accuracy']:.2f}% | {100 * run['best_validation_accuracy']:.2f}% | "
                     f"{metrics[-1]['paramCount']:,} | {sum(durations):.0f} min | {max(durations):.1f} min |")
    lines += ['', '![Phase 7 and Phase 5 seed 1 validation accuracy and parameters]'
              '(/assets/experiments/018-phase7-seed1-vs-phase5.png)', '',
              '### Seed 1 decisions', '',
              "Mean score is the selected action's mean search value. Widening shows how many widening "
              'candidates passed the output-preservation check and the best rank one reached (by mean score). '
              'The change after 20 epochs includes continued training; it is not an isolated effect of the action.', '',
              '| Generation ending | Validation before | Parameters | Selected action | Mean score | Candidates | Widening (best rank) | Rollouts | Tree depth | Search time | Change after training |',
              '| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    metrics, searches, _ = phase7
    accuracy = {row['globalEpoch'] // 20: row['valAcc'] for row in metrics}
    params = {row['globalEpoch'] // 20: row['paramCount'] for row in metrics}
    for search in searches:
        generation = search['generation']
        ranked = sorted(search['candidates'], key=lambda row: row['score'], reverse=True)
        chosen = next(row for row in ranked if row['action'] == search['actionChosen'])
        widen_ranks = [rank for rank, row in enumerate(ranked, 1) if 'Widen Conv' in row['action']]
        widening = f'{len(widen_ranks)} (#{widen_ranks[0]})' if widen_ranks else '0'
        change = 100 * (accuracy[generation + 1] - accuracy[generation])
        lines.append(f"| {generation} (epoch {20 * (generation + 1)}) | {100 * accuracy[generation]:.2f}% | "
                     f"{params[generation]:,} | {_action_label(search['actionChosen'])} | {chosen['score']:.4f} | "
                     f"{len(ranked)} | {widening} | {search['rollouts']} | {search['maxDepth']} | "
                     f"{search['durationSec'] / 60:.1f} min | {change:+.2f} pp |")
    lines += ['', 'The board files of this run mark the wrong candidate as `chosen` (a reporting bug in the '
              'fixed search, corrected afterwards: the board sorts candidate rows, and rows were matched to '
              'tree nodes by position). The executed action was always the highest mean score; this table '
              'uses the recorded executed action, not the flag.', '']
    lines += _phase7_seed1_findings(phase5, phase7)
    return lines


def _generations(numbers):
    numbers = list(numbers)
    if not numbers:
        return 'none'
    return ', '.join(str(n) for n in numbers[:-1]) + (' and ' if len(numbers) > 1 else '') + str(numbers[-1])


def _phase7_seed1_findings(phase5, phase7):
    """Findings computed from the saved seed-1 records, so they stay in step with the tables."""
    from documentation.website.scripts.generate_experiment_013_origin_action_scores import group_key
    (metrics5, searches5, run5), (metrics7, searches7, run7) = phase5, phase7
    minutes5 = [search['durationSec'] / 60 for search in searches5]
    minutes7 = [search['durationSec'] / 60 for search in searches7]
    deep = [s['generation'] for s in searches7 if s['maxDepth'] >= 2]
    single_visit = [s for s in searches7 if all(row['visits'] == 1 for row in s['candidates'])]
    widen = {s['generation']: [rank for rank, row in enumerate(
        sorted(s['candidates'], key=lambda row: row['score'], reverse=True), 1) if 'Widen Conv' in row['action']]
        for s in searches7}
    offered = [g for g, ranks in widen.items() if ranks]
    chosen_widen = [s['generation'] for s in searches7 if 'Widen Conv' in s['actionChosen']]
    gaps = []
    for s in searches7:
        top = sorted((row['score'] for row in s['candidates']), reverse=True)
        gaps.append(top[0] - top[1] if len(top) > 1 else 1.0)
    picked = {}
    for s in searches7:
        picked.setdefault(group_key({'action': s['actionChosen']}), []).append(s['generation'])
    weak = [f"{name.lower()} in generation{'s' if len(gens) > 1 else ''} {_generations(gens)}"
            for name, gens in picked.items() if name in ('Dropout', 'Sequential linear', 'Delete layer')]
    difference = 100 * (run7['final_validation_accuracy'] - run5['final_validation_accuracy'])
    lines = ['### What seed 1 shows', '',
             f'- **Search is about {sum(minutes5) / sum(minutes7):.0f} times cheaper in total.** Each search '
             f'took {min(minutes7):.1f}–{max(minutes7):.1f} minutes, against up to {max(minutes5):.0f} '
             'minutes in Phase 5, because each candidate costs one 5-epoch scoring run.']
    if deep:
        line = (f'- **Spare budget was used to search deeper.** In generations {_generations(deep)} the '
                'search covered the root actions and then expanded the best ones (tree depth 2).')
        if single_visit:
            counts = [len(s['candidates']) for s in single_visit]
            line += (f' In generations {_generations(s["generation"] for s in single_visit)} the '
                     f'{min(counts)}–{max(counts)} candidates used the whole budget at one visit each.')
        lines.append(line)
    if offered:
        ranks = [min(widen[g]) for g in offered]
        lines.append(
            f'- **Widening was offered in generations {_generations(offered)} '
            f'(best rank {min(ranks)}–{max(ranks)}) and '
            + ('never selected.**' if not chosen_widen else f'selected in generations {_generations(chosen_widen)}.**')
            + (f' From generation {offered[-1] + 1} onward no widening passed the output-preservation check: '
               'the widening code used by this run only handled a single path, and a residual convolution '
               'after `conv1` gives it two consumers. Fan-out widening, which handles that case, was added '
               'to `WidenConv` after this run.' if offered[-1] + 1 < len(searches7) else ''))
    lines.append(
        f'- **Close scores still decide.** The top two mean scores differed by less than 1 pp in '
        f'{sum(gap < 0.01 for gap in gaps)} of {len(gaps)} generations, within the noise of the '
        '1,120-example simulation-validation subset.'
        + (f' Weaker actions were selected: {"; ".join(weak)}.' if weak else '')
        + f' The model ended with {metrics7[-1]["paramCount"]:,} parameters against '
        f'{metrics5[-1]["paramCount"]:,} in Phase 5, and its final validation accuracy was '
        f'{abs(difference):.2f} pp {"lower" if difference < 0 else "higher"}.')
    return lines + ['']


def full_growth_score_fix_section():
    target = RUN / 'full_growth_5min_score_fix'
    lines = ['', '## Phase 7: full growth with the fixed Monte Carlo 2', '',
             'The Phase 5 recipe repeated with the fixed algorithm and the Phase 6 settings: convolution '
             'widening enabled, rollout depth 0 and 5 scoring epochs. All other settings match Phase 5, '
             'including the 5-minute soft search budget.', '',
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
    if (target / 'seed_1' / 'run.json').exists():
        lines += _phase7_seed1_analysis(target)
    return lines + ['', 'Run with `--phase full-score-fix`, then regenerate this report.', '']


def _caption_details(lines, first_figure=11):
    """Give every figure and table in the phase details a numbered caption, so each can be referenced.

    Figures continue the overview's numbering (from first_figure); a table caption names its phase and
    the subsection it belongs to. Existing captions are kept and renumbered."""
    out, figure, table, phase, section = [], first_figure, 0, '', ''
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith('### Phase'):
            phase, section = line.lstrip('#').strip(), ''
        elif line.startswith('####'):
            section = line.lstrip('#').strip()
        if line.startswith('!['):
            out.append(line)
            alt = line[2:line.index('](')]
            follow_at = next((i for i in range(index + 1, min(index + 3, len(lines))) if lines[i].strip()), None)
            if follow_at is not None and lines[follow_at].startswith('> [!CAPTION]'):
                lines[follow_at] = re.sub(r'Figure \d+\.', f'Figure {figure}.', lines[follow_at], count=1)
            else:
                out += ['', f'> [!CAPTION] Figure {figure}. {phase.split(":")[0]}: {alt}.']
            figure += 1
            index += 1
            continue
        if line.startswith('|') and (not out or not out[-1].startswith('|')):
            end = index
            while end < len(lines) and lines[end].startswith('|'):
                end += 1
            out += lines[index:end]
            table += 1
            follow = next((x for x in lines[end:end + 2] if x.strip()), '')
            if not follow.startswith('> [!CAPTION]'):
                where = f'{phase.split(":")[0]}: {section}' if section else phase
                out += ['', f'> [!CAPTION] Table {table}. {where}.']
            index = end
            continue
        out.append(line)
        index += 1
    return NEWLINE.join(out)


def write_details(lines):
    """Replace the report's phase-details section (between DETAILS_HEADING and SUMMARY_HEADING).

    The overview at the top of the report and the summary at its end are written by hand; this script
    owns only the detailed, phase-by-phase part, with every heading one level deeper."""
    details = _caption_details([('#' + line) if line.startswith('#') else line for line in NEWLINE.join(lines).split(NEWLINE)])
    text = REPORT.read_text(encoding='utf-8')
    start = text.index(DETAILS_HEADING)
    end = text.index(SUMMARY_HEADING, start)
    intro = 'Every phase in the order it was run, with its full protocol and tables.'
    parts = [text[:start] + DETAILS_HEADING, intro, details.strip(), text[end:]]
    REPORT.write_text((NEWLINE * 2).join(parts), encoding='utf-8')


def main():
    summary = json.loads((RUN / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['results']
    fig, ax = plt.subplots(figsize=(11, 6))
    for row in rows:
        vals = row['history']['val_acc']
        ax.plot(range(81, 81 + len(vals)), [100 * v for v in vals], label=LABELS[row['arm']],
                linewidth=2.5 if row['original_chosen'] else 1.5)
    ax.set(xlabel='Total training epoch', ylabel='Held-out validation accuracy (%)',
           title='Experiment 013: training after each generation-3 action')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, dpi=150)
    plt.close(fig)
    lines = ['## Phase 1: replaying the epoch-80 decision', '',
             'All arms start from the origin run generation-3 pre-decision checkpoint, after epoch 80. '
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
    lines += ['', '![Held-out validation accuracy for all replay arms](/assets/experiments/018-action-replay-validation.png)', '',
              'Full validation includes the original simulation-validation subset; held-out validation is the primary '
              'comparison. No test-set scores were used. The original run followed a different random trajectory, '
              'so its subsequent accuracy need not be reproduced exactly.', '']
    lines += scoring_section()
    lines += montecarlo2_section()
    lines += time_budget_section()
    lines += full_growth_section()
    from documentation.website.scripts.generate_experiment_013_seed1_analysis import generate as generate_seed1
    lines += generate_seed1()
    lines += score_fix_section()
    lines += full_growth_score_fix_section()
    lines += score_fix_20_epochs_section()
    lines += noise_fix_section()
    lines += full_grid_section()
    lines += simulation_fixes_after_phase10_section()
    lines += sim_fixes_check_section()
    lines += sim_fixes_growth_section()
    lines += decision_replay_section()
    lines += widening_growth_section()
    lines += scoring_calibration_section()
    lines += normalization_section()
    lines += long_run_section()
    lines += comparison_section()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_details(lines)
    print(f'Wrote {REPORT} and {CHART}')


if __name__ == '__main__':
    main()
