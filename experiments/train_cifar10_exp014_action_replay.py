"""Experiment 014: replay the Experiment 012 generation-3 decision and test Monte Carlo 2.

Run: python -u experiments/train_cifar10_exp014_action_replay.py --phase <name>

  replay              Phase 1: train every recorded action (and no action) from the checkpoint
  scoring             Phase 2: score each action with 1, 5, 10 and 20 scoring epochs
  montecarlo2         Phase 3: full Monte Carlo 2 search on the recorded actions
  time                Phase 4: Phase 3 with longer time budgets
  full                Phase 5: 10-generation growth from the Experiment 010 starter
  check-score-fix     Phase 6: Phase 3 with the fixed search, depth 0, 5 scoring epochs
  full-score-fix      Phase 7: Phase 5 with the Phase 6 settings and convolution widening
  check-score-fix-20  Phase 8: Phase 6 with 20 scoring epochs

Results are saved as each piece finishes, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.train_cifar10_exp008_generation_only import with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from experiments import train_cifar10_exp010_mcts_meta as exp010
from growingnn.actions.registry import generate_all_actions
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.gradient_descent import _evaluate, gradient_descent
from growingnn.utils.seed import seed_all
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_algorithms.montecarlo_alg import TreeNode
from growingnn.simulation.simulation_algorithms import montecarlo_alg_2
import growingnn.core.config as project_config

RUN = ROOT / 'experiments/output/train_cifar10/runs/exp012_family_search/v1/gen_10/family/cifar10/seed_1'
CHECKPOINT = RUN / 'board/decision_checkpoints/generation_3.pt'
BOARD = RUN / 'board/simulations/simulation_gen_3.json'
OUTPUT = ROOT / 'experiments/output/train_cifar10/runs/exp014_action_replay/v1'
SCORING_EPOCHS = (1, 5, 10, 20)
PHASE4_TIME_MINUTES = (20, 25, 30)


def source_files():
    return sorted((ROOT / 'growingnn').rglob('*.py')) + sorted((ROOT / 'experiments').glob('*.py'))


def source_fingerprint():
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(str(path.relative_to(ROOT)).encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_source_snapshot(target, expected):
    """Zip the source used by a run; refuse if it changed since *expected* was computed."""
    temporary = target.with_suffix('.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files():
            archive.write(path, path.relative_to(ROOT))
    if source_fingerprint() != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError('Source changed while creating the snapshot; retry with a stable worktree.')
    os.replace(temporary, target)


def heldout_validation_loader(val_loader, sim_val_loader):
    """Validation examples that the search never saw."""
    subset = sim_val_loader.dataset
    if not isinstance(subset, Subset) or subset.dataset is not val_loader.dataset:
        raise ValueError('Simulation validation must be a subset of validation dataset')
    used = set(subset.indices)
    remaining = [index for index in range(len(val_loader.dataset)) if index not in used]
    if not remaining:
        raise ValueError('No validation examples remain outside the search set')
    return DataLoader(Subset(val_loader.dataset, remaining),
                      batch_size=val_loader.batch_size, shuffle=False, num_workers=0)


SCORE_FIX_NOTE = ('Monte Carlo 2 score fixes (rollout values propagated as true means, mean-based '
                  'UCB exploitation, no root rollout) with convolution widening enabled, rollout '
                  'depth 0 and 5 scoring epochs.')
SCORE_FIX_SCORING_EPOCHS = 5
SCORE_FIX_MCTS_VALUES = {'MCTS_UCB1_C': 1.0, 'MCTS_UCB1_USE_SQRT': False,
                         'MCTS_PROPAGATE_ROLLOUT_VALUE': True, 'MCTS_ROLLOUT_DEPTH': 0,
                         'SIMULATION_MIN_ALGORITHM_ITERATION_RUNS': 3}


class WideningMonteCarlo2:
    """Monte Carlo 2 with convolution widening added to the generated actions."""
    @staticmethod
    def get_action(traced, config):
        config.ACTIONS_ENABLE_WIDEN_CONV = True
        return montecarlo_alg_2.get_action(traced, config)


FULL_GROWTH_PHASES = {
    'full': {'folder': 'full_growth_5min', 'phase_name': 'phase5', 'label': '014_full_growth',
             'name': 'Experiment 014 full growth', 'id_prefix': 'exp014', 'scoring_epochs': 20,
             'overrides': {}, 'widen_conv': False, 'note': None},
    'full-score-fix': {'folder': 'full_growth_5min_score_fix', 'phase_name': 'phase7',
                       'label': '014_full_growth_score_fix', 'name': 'Experiment 014 full growth (score fix)',
                       'id_prefix': 'exp014_score_fix', 'scoring_epochs': SCORE_FIX_SCORING_EPOCHS,
                       'overrides': SCORE_FIX_MCTS_VALUES, 'widen_conv': True, 'note': SCORE_FIX_NOTE},
}


def run_full_growth(output, seeds, phase='full'):
    """Phases 5 and 7: repeat Experiment 010's 10-generation recipe using Monte Carlo 2."""
    spec = FULL_GROWTH_PHASES[phase]
    target = output / spec['folder']
    digest = source_fingerprint()
    settings = dict(exp010.BASELINE, MCTS_UCB1_C=1.0, MCTS_UCB1_USE_SQRT=False,
        MCTS_PROPAGATE_ROLLOUT_VALUE=True, MCTS_ROLLOUT_DEPTH=1,
        MCTS_ROLLOUT_EPOCHS=1, MCTS_ROLLOUT_LR=0.0001,
        SIMULATION_MIN_ALGORITHM_ITERATION_RUNS=3, simulation_time=300.0,
        simulation_set_percent=10.0)
    settings.update(spec['overrides'])
    metadata = {'source_sha256': digest, 'dataset': 'cifar10', 'generations': 10,
                'epochs_per_generation': 20, 'scoring_epochs': spec['scoring_epochs'], 'settings': settings,
                'algorithm': 'montecarlo_alg_2', 'rollout_weight_decay': project_config.MCTS_ROLLOUT_WEIGHT_DECAY,
                'reference': 'Experiment 010 Pass 2 10-generation recipe'}
    if spec['widen_conv']:
        metadata['widen_conv'] = True
    if spec['note'] is not None:
        metadata['note'] = spec['note']
    manifest = target / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Full-growth settings or source changed; use a new output directory')
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_source_snapshot(target / 'source_snapshot.zip', digest)
        write_json(manifest, metadata)
    previous = exp010.montecarlo_alg
    exp010.montecarlo_alg = WideningMonteCarlo2 if spec['widen_conv'] else montecarlo_alg_2
    try:
        data = get_datasets('cifar10')
        for seed in seeds:
            print(f"START {spec['phase_name']} seed={seed}", flush=True)
            exp010.run_candidate('cifar10', seed, 'method', 'montecarlo2', settings, data,
                target / f'seed_{seed}', generations=10, phase_name=spec['phase_name'],
                experiment_label=spec['label'], experiment_name=spec['name'],
                experiment_id_prefix=spec['id_prefix'], scoring_epochs=spec['scoring_epochs'])
            if output == OUTPUT:
                from documentation.website.scripts.generate_experiment_014_report import main as generate_report
                generate_report()
    finally:
        exp010.montecarlo_alg = previous


class Phase3Board:
    """Capture the actual root statistics returned by Monte Carlo 2."""
    def __init__(self):
        self.simulation_metrics = {}
        self._current_generation = 3
        self.result = None

    def mcts_candidates_from_root(self, root, config):
        return [{'action': str(child.action), 'visits': child.visit_counter,
                 'score': child.value / child.visit_counter if child.visit_counter else None}
                for child in root.child_nodes]

    def mcts_search_tree_from_root(self, root, config, **kwargs):
        return None

    def on_simulation_finished(self, generation, **kwargs):
        self.result = kwargs


def run_montecarlo2_comparison(traced, actions, payload, device, seed, output, seconds,
                               epochs_values=SCORING_EPOCHS):
    """Run the complete Monte Carlo 2 search with the recorded root actions."""
    target = output / 'montecarlo2_epochs'
    metadata = {'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        'algorithm_sha256': hashlib.sha256(Path(montecarlo_alg_2.__file__).read_bytes()).hexdigest(),
        'seed': seed, 'simulation_subset_seed': 1, 'scoring_epochs': list(epochs_values),
        'simulation_time_sec': seconds, 'rollout_depth': project_config.MCTS_ROLLOUT_DEPTH,
        'rollout_weight_decay': project_config.MCTS_ROLLOUT_WEIGHT_DECAY,
        'ucb_c': project_config.MCTS_UCB1_C, 'ucb_sqrt': project_config.MCTS_UCB1_USE_SQRT,
        'propagate_rollout_value': project_config.MCTS_PROPAGATE_ROLLOUT_VALUE,
        'minimum_iterations': project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS,
        'device': str(device)}
    manifest = target / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Phase 3 settings or algorithm changed; use a new output directory')
    write_json(manifest, metadata)
    seed_all(1)
    data = get_datasets('cifar10')
    _, val, clean = data.loaders(128)
    sim_train, sim_val = sample_loaders(clean, val, round(len(clean.dataset) * .1), seed=1)
    original_generate = montecarlo_alg_2.generate_all_actions
    def generate(model, config):
        if model is traced:
            return [copy.deepcopy(action) for action, _ in actions]
        return original_generate(model, config)
    montecarlo_alg_2.generate_all_actions = generate
    try:
        for epochs in epochs_values:
            path = target / f'epochs_{epochs}.json'
            if path.exists():
                print(f'SKIP {path.name}', flush=True)
                continue
            board = Phase3Board()
            config = RunningConfig(generations=10, epochs=20,
                lr_scheduler=copy.deepcopy(payload['lr_scheduler']),
                simulation_scheduler=AlwaysSimulationScheduler(seconds, epochs),
                simulation_score=SimulationScore(weight_acc=1, weight_countW=0),
                criterion=nn.CrossEntropyLoss(label_smoothing=0.1), device=device,
                momentum=0.9, weight_decay=0.0001, max_grad_norm=1.0,
                seq_conv_insert_before_global_pool=True, enable_experiment_board=True,
                experiment_board=board)
            config.ACTIONS_ENABLE_WIDEN_CONV = True
            config.set_simulation_loaders(sim_train, sim_val)
            seed_all(seed)
            print(f'START Monte Carlo 2: scoring epochs={epochs}', flush=True)
            chosen, max_depth, rollouts = montecarlo_alg_2.get_action(traced, config)
            if board.result is None:
                raise RuntimeError('Monte Carlo 2 did not report its results')
            candidates = board.result['candidates']
            rows = []
            for index, (action, _) in enumerate(actions, 1):
                matches = [r for r in candidates if r['action'] == str(action)]
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one result for candidate {index}')
                rows.append({'arm': f'candidate_{index}', **matches[0]})
            write_json(path, {'epochs': epochs, 'selected_action': str(chosen) if chosen else None,
                'rollouts': rollouts, 'max_depth': max_depth,
                'duration_sec': board.result['duration_sec'], 'results': rows})
            print(f'DONE Monte Carlo 2: scoring epochs={epochs}, rollouts={rollouts}', flush=True)
        rows = [json.loads((target / f'epochs_{epochs}.json').read_text(encoding='utf-8'))
                for epochs in epochs_values]
        write_json(target / 'summary.json', {'manifest': metadata, 'runs': rows})
    finally:
        montecarlo_alg_2.generate_all_actions = original_generate


CHECK_PHASES = {
    'check-score-fix': {'label': 'Phase 6', 'folder': 'score_fix_check', 'scoring_epochs': SCORE_FIX_SCORING_EPOCHS},
    'check-score-fix-20': {'label': 'Phase 8', 'folder': 'score_fix_check_20ep', 'scoring_epochs': 20},
}


def run_score_fix_check(traced, actions, payload, device, seeds, output, seconds, phase='check-score-fix'):
    """Phases 6 and 8: rerun the Phase 3 search on the A08 decision with the score-fix settings."""
    spec = CHECK_PHASES[phase]
    label, scoring_epochs, target = spec['label'], spec['scoring_epochs'], output / spec['folder']
    previous = {name: getattr(project_config, name) for name in SCORE_FIX_MCTS_VALUES}
    for name, value in SCORE_FIX_MCTS_VALUES.items():
        setattr(project_config, name, value)
    try:
        for seed in seeds:
            print(f'START {label} check seed={seed}', flush=True)
            run_montecarlo2_comparison(traced, actions, payload, device, seed, target / f'seed_{seed}',
                                       seconds, epochs_values=(scoring_epochs,))
    finally:
        for name, value in previous.items():
            setattr(project_config, name, value)
    runs = []
    for seed in seeds:
        run = json.loads((target / f'seed_{seed}' / 'montecarlo2_epochs' /
                          f'epochs_{scoring_epochs}.json').read_text(encoding='utf-8'))
        runs.append({'seed': seed, **run})
        scores = ', '.join(f"{row['arm']}={row['score']:.4f}" for row in run['results'])
        print(f"{label} seed={seed}: selected {run['selected_action']} | {scores}", flush=True)
    write_json(target / 'summary.json', {'settings': SCORE_FIX_MCTS_VALUES,
                                         'scoring_epochs': scoring_epochs,
                                         'simulation_time_sec': seconds, 'runs': runs})
    print(f'{label.upper()} CHECK COMPLETE', flush=True)


def run_scoring_comparison(traced, actions, payload, device, seed, output):
    """Use the original TreeNode rollout and accuracy scoring, with depth zero."""
    target = output / 'scoring_epochs'
    metadata = {'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'seed': seed, 'simulation_subset_seed': 1, 'rollout_depth': 0,
                'scoring_epochs': list(SCORING_EPOCHS), 'device': str(device)}
    manifest = target / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Scoring output has different settings; select a new output directory')
    write_json(manifest, metadata)
    seed_all(1)
    data = get_datasets('cifar10')
    _, val, clean = data.loaders(128)
    sim_train, sim_val = sample_loaders(clean, val, round(len(clean.dataset) * .1), seed=1)
    previous_depth = project_config.MCTS_ROLLOUT_DEPTH
    project_config.MCTS_ROLLOUT_DEPTH = 0
    try:
        for epochs in SCORING_EPOCHS:
            for index, (action, original) in enumerate(actions, 1):
                path = target / f'epochs_{epochs}_candidate_{index}.json'
                if path.exists():
                    print(f'SKIP {path.name}', flush=True)
                    continue
                seed_all(seed)
                candidate = copy.deepcopy(traced)
                copy.deepcopy(action).execute(candidate)
                if candidate.param_count() != original['paramCount']:
                    raise ValueError('Candidate parameter count does not match recorded action')
                config = RunningConfig(generations=10, epochs=20,
                    lr_scheduler=copy.deepcopy(payload['lr_scheduler']),
                    simulation_scheduler=AlwaysSimulationScheduler(32, epochs),
                    simulation_score=SimulationScore(weight_acc=1, weight_countW=0),
                    criterion=nn.CrossEntropyLoss(label_smoothing=0.1), device=device,
                    momentum=0.9, weight_decay=0.0001, max_grad_norm=1.0)
                config.set_simulation_loaders(sim_train, sim_val)
                seed_all(seed)
                print(f'START scoring epochs={epochs} candidate={index}', flush=True)
                started = time.monotonic()
                node = TreeNode(None, action, candidate, config)
                try:
                    score = float(node.rollout())
                finally:
                    node.kill()
                write_json(path, {'epochs': epochs, 'arm': f'candidate_{index}',
                    'action': str(action), 'score': score,
                    'original_score': original['score'], 'seconds': time.monotonic() - started})
                print(f'DONE scoring epochs={epochs} candidate={index}: {score:.4f}', flush=True)
        rows = [json.loads((target / f'epochs_{epochs}_candidate_{index}.json').read_text(encoding='utf-8'))
                for epochs in SCORING_EPOCHS for index in range(1, 6)]
        write_json(target / 'summary.json', {'manifest': metadata, 'results': rows})
        print('SCORING COMPARISON COMPLETE', flush=True)
    finally:
        project_config.MCTS_ROLLOUT_DEPTH = previous_depth


def identity(text):
    return re.sub(r"(Widen Conv Action: \['[^']+',\s*\d+),\s*\d+(\])", r'\1\2', text)


def recorded_actions(traced, config, board):
    """Match recorded descriptions and restore the recorded widening seeds."""
    available = generate_all_actions(traced, config)
    result = []
    for row in board['candidates']:
        matches = [action for action in available if identity(str(action)) == identity(row['action'])]
        if len(matches) != 1:
            raise ValueError(f'Expected one exact match for {row["action"]}, found {len(matches)}')
        action = copy.deepcopy(matches[0])
        if isinstance(action, WidenConv):
            match = re.search(r"Widen Conv Action: \['[^']+',\s*\d+,\s*(\d+)\]", row['action'])
            if not match:
                raise ValueError('Recorded widening seed missing')
            action.params[2] = int(match.group(1))
        if str(action) != row['action']:
            raise ValueError(f'Reconstructed action differs from recorded action: {action}')
        result.append((action, row))
    if len(result) != 5:
        raise ValueError(f'Expected 5 original candidates, got {len(result)}')
    return result


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--phase', choices=('replay', 'scoring', 'montecarlo2', 'time',
                                            *FULL_GROWTH_PHASES, *CHECK_PHASES), default='replay')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--simulation-time', type=float, default=32.0)
    args = parser.parse_args()
    if args.epochs < 1 or args.simulation_time < 0:
        parser.error('--epochs must be positive and --simulation-time nonnegative')
    if args.phase in FULL_GROWTH_PHASES:
        run_full_growth(args.output, args.seeds, args.phase)
        return
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    board = json.loads(BOARD.read_text(encoding='utf-8'))
    traced = TracedModel.create(payload['model'], payload['input_shape'])
    if payload['generation'] != 3 or traced.param_count() != board['paramCountBefore']:
        raise ValueError('Unexpected checkpoint or parameter count')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = RunningConfig(generations=10, epochs=args.epochs,
                           lr_scheduler=payload['lr_scheduler'],
                           criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
                           device=device, momentum=0.9, weight_decay=0.0001,
                           max_grad_norm=1.0, seq_conv_insert_before_global_pool=True)
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    seed_all(args.seed)
    actions = recorded_actions(traced, config, board)
    if args.phase == 'time':
        baseline_path = args.output / 'montecarlo2_epochs' / 'manifest.json'
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        expected = {'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    'algorithm_sha256': hashlib.sha256(Path(montecarlo_alg_2.__file__).read_bytes()).hexdigest(),
                    'seed': args.seed, 'device': str(device),
                    'rollout_depth': project_config.MCTS_ROLLOUT_DEPTH,
                    'rollout_weight_decay': project_config.MCTS_ROLLOUT_WEIGHT_DECAY,
                    'ucb_c': project_config.MCTS_UCB1_C, 'ucb_sqrt': project_config.MCTS_UCB1_USE_SQRT,
                    'propagate_rollout_value': project_config.MCTS_PROPAGATE_ROLLOUT_VALUE,
                    'minimum_iterations': project_config.SIMULATION_MIN_ALGORITHM_ITERATION_RUNS}
        if any(baseline.get(key) != value for key, value in expected.items()):
            raise ValueError('Phase 3 baseline settings or algorithm differ from Phase 4')
        for minutes in PHASE4_TIME_MINUTES:
            print(f'START Phase 4: {minutes}-minute budget', flush=True)
            run_montecarlo2_comparison(traced, actions, payload, device, args.seed,
                args.output / 'montecarlo2_time' / f'{minutes}min', minutes * 60,
                epochs_values=(20,))
        if args.output == OUTPUT:
            from documentation.website.scripts.generate_experiment_014_report import main as generate_report
            generate_report()
        return
    if args.phase in CHECK_PHASES:
        run_score_fix_check(traced, actions, payload, device, args.seeds, args.output, args.simulation_time,
                            args.phase)
        return
    if args.phase == 'montecarlo2':
        run_montecarlo2_comparison(traced, actions, payload, device, args.seed, args.output, args.simulation_time)
        return
    if args.phase == 'scoring':
        run_scoring_comparison(traced, actions, payload, device, args.seed, args.output)
        return
    metadata = {'checkpoint': str(CHECKPOINT), 'checkpoint_sha256': hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                'source_board': str(BOARD), 'decision_generation': 3, 'decision_epoch': 80,
                'seed': args.seed, 'epochs': args.epochs, 'device': str(device),
                'simulation_subset_seed': 1,
                'note': 'Same nominal seed per arm; different architectures may consume randomness differently.'}
    manifest = args.output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != metadata:
        raise ValueError('Output belongs to different settings; choose a new output directory')
    write_json(manifest, metadata)
    seed_all(1)
    data = get_datasets('cifar10')
    # The held-out validation set excludes the subset used by the original search.
    _, original_val, clean = data.loaders(128)
    _, sim_val = sample_loaders(clean, original_val, round(len(clean.dataset) * 0.1), seed=1)
    heldout = heldout_validation_loader(original_val, sim_val)
    print(f'Held-out validation examples: {len(heldout.dataset)}', flush=True)
    arms = [(None, None)] + actions
    for index, (action, row) in enumerate(arms):
        name = 'control' if action is None else f'candidate_{index}'
        result_path = args.output / f'{name}.json'
        if result_path.exists():
            print(f'SKIP {name}', flush=True)
            continue
        seed_all(args.seed)
        candidate = copy.deepcopy(traced)
        if action is not None:
            copy.deepcopy(action).execute(candidate)
            if candidate.param_count() != row['paramCount']:
                raise ValueError(f'Parameter mismatch for {name}')
        before_loss, before = _evaluate(candidate.gm.to(device), heldout, config.criterion, device)
        # Reset after construction so the data loader starts from the same seed.
        seed_all(args.seed)
        train, full_val, _ = data.loaders(128)
        train = with_cutout(train)
        scheduler = copy.deepcopy(payload['lr_scheduler'])
        print(f'START {name}: {str(action) if action else "unchanged"}', flush=True)
        started = time.monotonic()
        model, history = gradient_descent(candidate.gm, args.epochs, train, heldout,
                                         config.criterion, scheduler, quiet=True,
                                         momentum=config.momentum, weight_decay=config.weight_decay,
                                         device=device, max_grad_norm=config.max_grad_norm)
        full_loss, full_accuracy = _evaluate(model, full_val, config.criterion, device)
        result = {'arm': name, 'action': str(action) if action else 'unchanged',
                  'original_simulation_score': row['score'] if row else None,
                  'original_visits': row['visits'] if row else None,
                  'original_chosen': row['chosen'] if row else False,
                  'parameter_count': candidate.param_count(),
                  'heldout_validation_size': len(heldout.dataset),
                  'initial_heldout_accuracy': before, 'initial_heldout_loss': before_loss,
                  'final_full_validation_accuracy': full_accuracy,
                  'final_full_validation_loss': full_loss,
                  'training_seconds': time.monotonic() - started, 'history': history}
        write_json(result_path, result)
        print(f'DONE {name}: heldout={history["val_acc"][-1]:.4f} full={full_accuracy:.4f}', flush=True)
    results = [json.loads((args.output / ('control.json' if i == 0 else f'candidate_{i}.json')).read_text(encoding='utf-8'))
               for i in range(len(arms))]
    baseline = results[0]['history']['val_acc'][-1]
    for result in results:
        result['final_heldout_accuracy'] = result['history']['val_acc'][-1]
        result['gain_vs_control_pp'] = 100 * (result['final_heldout_accuracy'] - baseline)
    write_json(args.output / 'summary.json', {'manifest': metadata, 'results': results})
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
