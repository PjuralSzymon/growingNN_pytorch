"""Create Experiment 015's report and learning-curve figure from saved results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1'
REPORT = ROOT / 'documentation/website/content/experiments/experiment-015-action-replay.md'
CHART = ROOT / 'documentation/website/app/public/assets/experiments/015-action-replay-validation.png'
SCORING_CHART = ROOT / 'documentation/website/app/public/assets/experiments/015-scoring-epochs.png'
MCTS_CHART = ROOT / 'documentation/website/app/public/assets/experiments/015-montecarlo2-epochs.png'
TIME_CHART = ROOT / 'documentation/website/app/public/assets/experiments/015-montecarlo2-time.png'
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
               title='Experiment 15: effect of scoring epochs', xticks=epochs_list)
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
        lines += ['', '![Simulation accuracy by scoring epochs and action](/assets/experiments/015-scoring-epochs.png)']
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
               title='Experiment 15: Monte Carlo 2 search values', xticks=sorted(plotted))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        MCTS_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(MCTS_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by scoring epochs](/assets/experiments/015-montecarlo2-epochs.png)']
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
               title='Experiment 15: Monte Carlo 2 time budgets',
               xticks=list(positions), xticklabels=[label for label, _ in plotted])
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        TIME_CHART.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(TIME_CHART, dpi=150)
        plt.close(fig)
        lines += ['', '![Monte Carlo 2 search values by time budget](/assets/experiments/015-montecarlo2-time.png)']
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
             'This phase repeats the Phase 3 search on the same Experiment 013 generation-3 checkpoint '
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
    else:
        lines += ['', 'Run with `--phase full-grid --seeds 1` (all four cells), then regenerate this report.']
    return lines + ['']


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


PHASE7_CHART = ROOT / 'documentation/website/app/public/assets/experiments/015-phase7-seed1-vs-phase5.png'


def _load_growth_run(directory):
    """Per-epoch metrics and per-generation searches of one full-growth seed."""
    board = directory / 'board'
    metrics = json.loads((board / 'metrics/training.json').read_text(encoding='utf-8'))['epochs']
    searches = [json.loads(path.read_text(encoding='utf-8')) for path in sorted(
        (board / 'simulations').glob('simulation_gen_*.json'), key=lambda path: int(path.stem.rsplit('_', 1)[1]))]
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    return metrics, searches, run


def _action_label(action):
    from documentation.website.scripts.generate_experiment_013_action_scores import candidate_detail, group_key
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
        size.step(epochs, [row['paramCount'] for row in metrics], where='post', label=name, color=color)
    acc.set(ylabel='Validation accuracy (%)', ylim=(0, 100), title='Experiment 15 seed 1: Phase 7 vs Phase 5')
    size.set(xlabel='Training epoch', ylabel='Parameters')
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
              '(/assets/experiments/015-phase7-seed1-vs-phase5.png)', '',
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
    from documentation.website.scripts.generate_experiment_013_action_scores import group_key
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


def main():
    summary = json.loads((RUN / 'summary.json').read_text(encoding='utf-8'))
    rows = summary['results']
    fig, ax = plt.subplots(figsize=(11, 6))
    for row in rows:
        vals = row['history']['val_acc']
        ax.plot(range(81, 81 + len(vals)), [100 * v for v in vals], label=LABELS[row['arm']],
                linewidth=2.5 if row['original_chosen'] else 1.5)
    ax.set(xlabel='Total training epoch', ylabel='Held-out validation accuracy (%)',
           title='Experiment 15: training after each generation-3 action')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    CHART.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART, dpi=150)
    plt.close(fig)
    lines = ['---', 'title: "Experiment 15: replaying the epoch-80 decision"', '---', '',
             'All arms start from the Experiment 13 generation-3 pre-decision checkpoint, after epoch 80. '
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
    lines += ['', '![Held-out validation accuracy for all replay arms](/assets/experiments/015-action-replay-validation.png)', '',
              'Full validation includes the original simulation-validation subset; held-out validation is the primary '
              'comparison. No test-set scores were used. The original run followed a different random trajectory, '
              'so its subsequent accuracy need not be reproduced exactly.', '']
    lines += scoring_section()
    lines += montecarlo2_section()
    lines += time_budget_section()
    lines += full_growth_section()
    from documentation.website.scripts.generate_experiment_015_seed1_analysis import generate as generate_seed1
    lines += generate_seed1()
    lines += score_fix_section()
    lines += full_growth_score_fix_section()
    lines += score_fix_20_epochs_section()
    lines += noise_fix_section()
    lines += full_grid_section()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote {REPORT} and {CHART}')


if __name__ == '__main__':
    main()
