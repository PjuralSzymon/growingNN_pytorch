"""Experiment 015 (code IDs keep "014"): replay the Experiment 013 generation-3 decision and test Monte Carlo 2.

Run: python -u experiments/train_cifar10_exp014_action_replay.py --phase <name>

  replay              Phase 1: train every recorded action (and no action) from the checkpoint
  scoring             Phase 2: score each action with 1, 5, 10 and 20 scoring epochs
  montecarlo2         Phase 3: full Monte Carlo 2 search on the recorded actions
  time                Phase 4: Phase 3 with longer time budgets
  full                Phase 5: 10-generation growth from the Experiment 010 starter
  check-score-fix     Phase 6: Phase 3 with the fixed search, depth 0, 5 scoring epochs
  full-score-fix      Phase 7: Phase 5 with the Phase 6 settings and convolution widening
  check-score-fix-20  Phase 8: Phase 6 with 20 scoring epochs
  full-grid           Phase 10: Phase 7 growth with the Phase 9 search, UCB C x simulation set grid
  full-sim-fixes      Phase 12: the best Phase 10 cell (C = 1.0, CCS) with the simulation fixes made after Phase 10
  decision-replay     Phase 13: train every growing candidate of the Phase 12 seed-2 generation-3 search for
                      a full generation
  full-widening       Phase 14: Phase 12 with widening offered for every convolution (run with --seeds 1 2 3)
  full-normalization  Phase 16: short Phase 14 runs (5 generations, --seeds 1): a) with the add-normalization
                      action, b) the same from the starter with BatchNorm after conv1
  full-residual-stem  Phase 16c: Phase 16b from a starter with a residual block relu(x + BN(conv(x))) (--seeds 1)
  scoring-calibration Phase 15a: score the Phase 13 arms with longer scoring and larger simulation sets,
                      3 measurements each, and compare with their real 20-epoch result
  check-noise-fix     Phase 9: Phase 6 with a 5-minute budget, type-balanced expansion and
                      follow-ups, one training seed per search and a 2-epoch score average

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

from experiments.train_cifar10_exp008_generation_only import CosineDecayScheduler, with_cutout
from experiments.train_cifar10_exp009_best_reference import get_datasets
from experiments import train_cifar10_exp010_mcts_meta as exp010
from growingnn.actions.registry import generate_all_actions
from growingnn.actions.add_seq_dropout_layer import AddSeqDropoutLayer
from growingnn.actions.delete_layer import DelLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.simulation.simulation_set import sample_loaders
from growingnn.training.gradient_descent import _evaluate, gradient_descent
from growingnn.training.trainer import train_generations
from growingnn.utils.seed import seed_all
from growingnn.simulation.score_functions.simulation_score import SimulationScore
from growingnn.simulation.simulation_schedulers import AlwaysSimulationScheduler
from growingnn.simulation.simulation_algorithms.montecarlo_alg import TreeNode
from growingnn.simulation.simulation_algorithms import montecarlo_alg_2
from growingnn.simulation.simulation_sets import CcsSimulationSet, ModerateSimulationSet
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


class GridMonteCarlo2:
    """Phase 10 search: Phase 9 settings plus a time budget that grows with the number of candidates."""

    def __init__(self, seconds_per_candidate, minimum_seconds, score_last_epochs):
        self.seconds_per_candidate = seconds_per_candidate
        self.minimum_seconds = minimum_seconds
        self.score_last_epochs = score_last_epochs

    def get_action(self, traced, config):
        config.ACTIONS_ENABLE_WIDEN_CONV = True
        config.simulation_score.average_last_epochs = self.score_last_epochs
        candidates = len(generate_all_actions(traced, config))
        config.simulation_scheduler.simulation_time = max(self.minimum_seconds,
                                                          self.seconds_per_candidate * candidates)
        return montecarlo_alg_2.get_action(traced, config)


GRID_UCB_C = (0.05, 1.0)
GRID_SIMULATION_SETS = {'moderate': ModerateSimulationSet, 'ccs': CcsSimulationSet}
GRID_SECONDS_PER_CANDIDATE = 24.0  # about two 5-epoch rollouts per candidate (Phase 9: 10-12 s each)
GRID_MINIMUM_SECONDS = 300.0
GRID_NOTE = ('Phase 10: Phase 7 growth with the Phase 9 search (type-balanced expansion, one training seed '
             'per search, 2-epoch score average), convolution widening with fan-out, a budget of 24 s per '
             'candidate (at least 5 minutes), and a grid over UCB C and the simulation-set selector.')


def grid_spec(ucb_c, simulation_set):
    """One Phase 10 cell: a UCB C value and a simulation-set selector."""
    cell = f"c{str(ucb_c).replace('.', 'p')}_{simulation_set}"
    return {'folder': f'full_grid/{cell}', 'phase_name': 'phase10', 'label': '014_full_grid',
            'name': f'Experiment 015 Phase 10 ({cell})', 'id_prefix': f'exp014_grid_{cell}',
            'scoring_epochs': SCORE_FIX_SCORING_EPOCHS,
            'overrides': {**SCORE_FIX_MCTS_VALUES, 'MCTS_UCB1_C': ucb_c},
            'widen_conv': True, 'note': GRID_NOTE, 'simulation_set': simulation_set,
            'search': GridMonteCarlo2(GRID_SECONDS_PER_CANDIDATE, GRID_MINIMUM_SECONDS, score_last_epochs=2),
            'search_budget': {'seconds_per_candidate': GRID_SECONDS_PER_CANDIDATE,
                              'minimum_seconds': GRID_MINIMUM_SECONDS, 'score_last_epochs': 2}}


SIM_FIXES_NOTE = ('Phase 12: the Phase 10 cell C = 1.0 with CCS, run with the simulation fixes made after Phase 10 '
                  '(lazy expansion, restarted scoring learning rate, futures continuing from trained models, '
                  'measuring twice before expanding).')


def sim_fixes_spec():
    """Phase 12: the best Phase 10 cell (C = 1.0, CCS) with the simulation fixes made after Phase 10."""
    return {**grid_spec(1.0, 'ccs'), 'folder': 'full_growth_sim_fixes', 'phase_name': 'phase12',
            'label': '014_full_growth_sim_fixes', 'name': 'Experiment 015 Phase 12 (simulation fixes)',
            'id_prefix': 'exp014_sim_fixes', 'note': SIM_FIXES_NOTE}


def full_growth_settings(spec):
    settings = dict(exp010.BASELINE, MCTS_UCB1_C=1.0, MCTS_UCB1_USE_SQRT=False,
        MCTS_PROPAGATE_ROLLOUT_VALUE=True, MCTS_ROLLOUT_DEPTH=1,
        MCTS_ROLLOUT_EPOCHS=1, MCTS_ROLLOUT_LR=0.0001,
        SIMULATION_MIN_ALGORITHM_ITERATION_RUNS=3, simulation_time=300.0,
        simulation_set_percent=10.0)
    settings.update(spec['overrides'])
    return settings


WIDENING_NOTE = ('Phase 14: Phase 12 rerun after the widening refactor (91085d0): every convolution can be '
                 'widened, new channels start with random weights and zero weights where they are read.')


def widening_spec():
    """Phase 14: Phase 12 with widening offered for every convolution."""
    return {**sim_fixes_spec(), 'folder': 'full_growth_widening', 'phase_name': 'phase14',
            'label': '014_full_growth_widening', 'name': 'Experiment 015 Phase 14 (widening for every convolution)',
            'id_prefix': 'exp014_widening', 'note': WIDENING_NOTE}


NORMALIZATION_GENERATIONS = 5
NORMALIZATION_NOTE = ('Phase 16: a short Phase 14 run with the add-normalization action (541cab3): a BatchNorm '
                      'after any convolution or linear layer that has none, with default initialization. '
                      f'{NORMALIZATION_GENERATIONS} generations, to see how the simulation scores it.')


class NormalizationMonteCarlo2(GridMonteCarlo2):
    """The Phase 12/14 search with the add-normalization action among the candidates."""

    def get_action(self, traced, config):
        config.ACTIONS_ENABLE_ADD_NORMALIZATION = True
        return super().get_action(traced, config)


def normalization_spec():
    """Phase 16: Phase 14 with normalization candidates, shortened to a few generations."""
    return {**widening_spec(), 'folder': 'full_growth_normalization', 'phase_name': 'phase16',
            'label': '014_full_growth_normalization', 'name': 'Experiment 015 Phase 16 (add normalization)',
            'id_prefix': 'exp014_normalization', 'note': NORMALIZATION_NOTE,
            'generations': NORMALIZATION_GENERATIONS, 'add_normalization': True,
            'search': NormalizationMonteCarlo2(GRID_SECONDS_PER_CANDIDATE, GRID_MINIMUM_SECONDS,
                                               score_last_epochs=2)}


class NormalizedStemStarter(nn.Module):
    """The Experiment 010 starter with a BatchNorm between conv1 and its ReLU, as in the residual starter's stem.

    Same width (8 channels), no residual block: it isolates what the BatchNorm alone changes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.linear2 = nn.Linear(8, 10, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


NORMALIZED_STEM_NOTE = ('Phase 16b: Phase 16 started from the Experiment 010 starter with a BatchNorm after conv1 '
                        '(conv1 -> BatchNorm -> ReLU, as in the residual starter of the other branch), without its '
                        'residual block or wider stem. Does BatchNorm alone close the gap, or is the residual '
                        'connection needed as well?')


def normalized_stem_spec():
    """Phase 16b: Phase 16 from a starter that already has BatchNorm in its stem."""
    return {**normalization_spec(), 'folder': 'full_growth_normalized_stem', 'phase_name': 'phase16b',
            'label': '014_full_growth_normalized_stem', 'name': 'Experiment 015 Phase 16b (BatchNorm in the stem)',
            'id_prefix': 'exp014_normalized_stem', 'note': NORMALIZED_STEM_NOTE,
            'starter': NormalizedStemStarter, 'starter_name': 'TinyCifar10Starter + BatchNorm after conv1'}



class ResidualStemStarter(NormalizedStemStarter):
    """The Phase 16b starter plus the residual block of the other branch's starter: relu(x + BN(conv(x))).

    Same width (8 channels) as Phase 16b, so the run isolates what the residual block adds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.block_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.block_bn = nn.BatchNorm2d(8)

    def forward(self, x):
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.relu(x + self.block_bn(self.block_conv(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


RESIDUAL_STEM_NOTE = ('Phase 16c: Phase 16b with the residual block of the other branch starter after the stem: '
                      'relu(x + BatchNorm(conv(x))), still 8 channels. Does the residual connection close the gap?')


def residual_stem_spec():
    """Phase 16c: Phase 16b from a starter with a residual block after its stem."""
    return {**normalized_stem_spec(), 'folder': 'full_growth_residual_stem', 'phase_name': 'phase16c',
            'label': '014_full_growth_residual_stem', 'name': 'Experiment 015 Phase 16c (residual block)',
            'id_prefix': 'exp014_residual_stem', 'note': RESIDUAL_STEM_NOTE, 'starter': ResidualStemStarter,
            'starter_name': 'TinyCifar10Starter + BatchNorm after conv1 + residual block relu(x + BN(conv(x)))'}

def run_full_growth(output, seeds, phase='full', spec=None):
    """Phases 5, 7 and 10: repeat Experiment 010's 10-generation recipe using Monte Carlo 2."""
    spec = spec or FULL_GROWTH_PHASES[phase]
    target = output / spec['folder']
    digest = source_fingerprint()
    settings = full_growth_settings(spec)
    generations = spec.get('generations', 10)
    metadata = {'source_sha256': digest, 'dataset': 'cifar10', 'generations': generations,
                'epochs_per_generation': 20, 'scoring_epochs': spec['scoring_epochs'], 'settings': settings,
                'algorithm': 'montecarlo_alg_2', 'rollout_weight_decay': project_config.MCTS_ROLLOUT_WEIGHT_DECAY,
                'reference': 'Experiment 010 Pass 2 10-generation recipe'}
    if spec['widen_conv']:
        metadata['widen_conv'] = True
    if spec['note'] is not None:
        metadata['note'] = spec['note']
    for key in ('simulation_set', 'search_budget', 'add_normalization', 'starter_name'):
        if key in spec:
            metadata[key] = spec[key]
    manifest = target / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Full-growth settings or source changed; use a new output directory')
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_source_snapshot(target / 'source_snapshot.zip', digest)
        write_json(manifest, metadata)
    previous, previous_sampler, previous_starter = exp010.montecarlo_alg, exp010.sample_loaders, exp010.initial_graph
    exp010.montecarlo_alg = spec.get('search') or (WideningMonteCarlo2 if spec['widen_conv'] else montecarlo_alg_2)
    if 'simulation_set' in spec:
        exp010.sample_loaders = GRID_SIMULATION_SETS[spec['simulation_set']]().sample_loaders
    if 'starter' in spec:
        exp010.initial_graph = lambda dataset='cifar10': spec['starter']()
    try:
        data = get_datasets('cifar10')
        for seed in seeds:
            print(f"START {spec['phase_name']} seed={seed}", flush=True)
            exp010.run_candidate('cifar10', seed, 'method', 'montecarlo2', settings, data,
                target / f'seed_{seed}', generations=generations, phase_name=spec['phase_name'],
                experiment_label=spec['label'], experiment_name=spec['name'],
                experiment_id_prefix=spec['id_prefix'], scoring_epochs=spec['scoring_epochs'])
            if output == OUTPUT:
                from documentation.website.scripts.generate_experiment_015_report import main as generate_report
                generate_report()
    finally:
        exp010.montecarlo_alg, exp010.sample_loaders, exp010.initial_graph = previous, previous_sampler, previous_starter


DECISION_REPLAY_RUN = OUTPUT / 'full_growth_sim_fixes' / 'seed_2'
DECISION_REPLAY_GENERATION = 3
DECISION_REPLAY_INPUT_SHAPE = (1, 3, 32, 32)
DECISION_REPLAY_SKIPPED = (AddSeqDropoutLayer, DelLayer, DelNeurons)


class RecordedChoices:
    """Stands in for the search and applies the actions a full-growth run selected, in order."""

    def __init__(self, choices):
        self.choices = list(choices)

    def get_action(self, traced, config):
        if not self.choices:
            return None, None, None
        return recorded_action(generate_all_actions(traced, config), self.choices.pop(0)), None, None


def decision_replay_config(device, lr_scheduler, generations=10, simulation_alg=None):
    """The Phase 12 training and action settings."""
    config = RunningConfig(generations=generations, epochs=exp010.EPOCHS, lr_scheduler=lr_scheduler,
                           simulation_alg=simulation_alg,
                           simulation_scheduler=AlwaysSimulationScheduler(0.0, SCORE_FIX_SCORING_EPOCHS),
                           criterion=nn.CrossEntropyLoss(label_smoothing=0.1), device=device,
                           momentum=0.9, weight_decay=0.0001, max_grad_norm=1.0,
                           seq_conv_insert_before_global_pool=True, quiet=True)
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    return config


def decision_architecture(searches, generation, config):
    """The starter with the run's choices before *generation* applied, untrained."""
    traced = TracedModel.create(exp010.initial_graph('cifar10'), DECISION_REPLAY_INPUT_SHAPE)
    for search in searches[:generation]:
        recorded_action(generate_all_actions(traced, config), search['actionChosen']).execute(traced)
    return traced


def train_decision_model(searches, generation, seed, data, device):
    """Grow the run's starter again with its recorded choices and stop where the search at *generation* ran.

    The run saved no checkpoints, so the architecture and training recipe match the run but the weights
    do not: the original searches consumed random numbers between generations.
    """
    seed_all(seed)
    train, val, _ = data.loaders(128)
    config = decision_replay_config(device, CosineDecayScheduler(total_epochs=10 * exp010.EPOCHS),
                                    generations=generation + 1, simulation_alg=RecordedChoices(
                                        search['actionChosen'] for search in searches[:generation]))
    model, history = train_generations(exp010.initial_graph('cifar10'), with_cutout(train), val, config,
                                       sim_train_loader=train, sim_val_loader=val)
    return model.state_dict(), config.lr_scheduler, history['val_acc'][-1]


def run_decision_replay(output, epochs):
    """Phase 13: train every growing candidate of one Phase 12 search for a full generation."""
    generation = DECISION_REPLAY_GENERATION
    target = output / 'decision_replay' / f'gen_{generation}'
    digest = source_fingerprint()
    metadata = {'source_sha256': digest, 'run': Path(os.path.relpath(DECISION_REPLAY_RUN, output)).as_posix(),
                'generation': generation, 'epochs': epochs,
                'skipped_types': [kind.__name__ for kind in DECISION_REPLAY_SKIPPED]}
    manifest = target / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Decision-replay settings or source changed; use a new output directory')
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_source_snapshot(target / 'source_snapshot.zip', digest)
        write_json(manifest, metadata)
    exp010.apply_values(full_growth_settings(sim_fixes_spec()))
    searches = [json.loads((DECISION_REPLAY_RUN / f'board/simulations/simulation_gen_{index}.json')
                           .read_text(encoding='utf-8')) for index in range(generation + 1)]
    search = searches[generation]
    seed = json.loads((DECISION_REPLAY_RUN / 'run.json').read_text(encoding='utf-8'))['seed']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = get_datasets('cifar10')
    checkpoint = target / 'decision_model.pt'
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    else:
        print(f'START rebuilding the generation-{generation} model', flush=True)
        state, lr_scheduler, val_acc = train_decision_model(searches, generation, seed, data, device)
        payload = {'state_dict': state, 'lr_scheduler': lr_scheduler, 'validation_accuracy': val_acc}
        torch.save(payload, checkpoint)
    config = decision_replay_config(device, payload['lr_scheduler'])
    traced = decision_architecture(searches, generation, config)
    traced.gm.load_state_dict(payload['state_dict'])
    if traced.param_count() != search['paramCountBefore']:
        raise ValueError('Rebuilt model differs from the recorded architecture')
    available = generate_all_actions(traced, config)
    arms = [(None, None)]
    for row in search['candidates']:
        action = recorded_action(available, row['action'])
        if not isinstance(action, DECISION_REPLAY_SKIPPED):
            arms.append((action, row))
    for index, (action, row) in enumerate(arms):
        name = 'control' if action is None else f'candidate_{index:02d}'
        result_path = target / f'{name}.json'
        if result_path.exists():
            print(f'SKIP {name}', flush=True)
            continue
        candidate = copy.deepcopy(traced)
        if action is not None:
            copy.deepcopy(action).execute(candidate)
            if candidate.param_count() != row['paramCount']:
                raise ValueError(f'Parameter mismatch for {name}')
        seed_all(seed)
        train, val, _ = data.loaders(128)
        print(f'START {name}: {str(action) if action else "unchanged"}', flush=True)
        started = time.monotonic()
        _, history = gradient_descent(candidate.gm, epochs, with_cutout(train), val, config.criterion,
                                      copy.deepcopy(config.lr_scheduler), quiet=True, momentum=config.momentum,
                                      weight_decay=config.weight_decay, device=device,
                                      max_grad_norm=config.max_grad_norm)
        write_json(result_path, {'arm': name, 'action': str(action) if action else 'unchanged',
                                 'type': type(action).__name__ if action else None,
                                 'simulation_score': row['score'] if row else None,
                                 'simulation_visits': row['visits'] if row else None,
                                 'chosen': bool(row and row['chosen']), 'parameter_count': candidate.param_count(),
                                 'final_validation_accuracy': history['val_acc'][-1],
                                 'training_seconds': time.monotonic() - started, 'history': history})
        print(f"DONE {name}: validation={history['val_acc'][-1]:.4f}", flush=True)
    results = [json.loads((target / ('control.json' if index == 0 else f'candidate_{index:02d}.json'))
                          .read_text(encoding='utf-8')) for index in range(len(arms))]
    write_json(target / 'summary.json', {'manifest': metadata, 'rebuilt_validation_accuracy':
                                         payload['validation_accuracy'], 'results': results})
    if output == OUTPUT:
        from documentation.website.scripts.generate_experiment_015_report import main as generate_report
        generate_report()
    print('COMPLETE', flush=True)


CALIBRATION_SETTINGS = ((5, 10.0), (10, 10.0), (5, 25.0), (10, 25.0), (15, 25.0))  # (scoring epochs, % of train)
CALIBRATION_MEASUREMENTS = 3
CALIBRATION_SCORE_LAST_EPOCHS = 2
CALIBRATION_SEARCH_SEED = 2026  # measurement i of every arm uses seed + i, as within one search
CALIBRATION_TOP = 5


def calibration_setting_name(epochs, percent):
    return f'epochs_{epochs}_set_{percent:g}'


def _spearman(first, second):
    from scipy.stats import spearmanr
    return float(spearmanr(first, second).statistic)


def calibration_metrics(rows, control_accuracy, top=CALIBRATION_TOP):
    """Compare one scoring setting with real training; *rows* are the candidates (control excluded)."""
    real = [row['real_validation_accuracy'] for row in rows]
    means = [sum(row['scores']) / len(row['scores']) for row in rows]
    by_mean = sorted(range(len(rows)), key=lambda index: -means[index])
    by_real = sorted(range(len(rows)), key=lambda index: -real[index])
    chosen, best = by_mean[0], by_real[0]
    spreads = [max(row['scores']) - min(row['scores']) for row in rows]
    return {'spearman_one_measurement': _spearman([row['scores'][0] for row in rows], real),
            'spearman_mean': _spearman(means, real),
            'chosen': rows[chosen]['arm'], 'chosen_real_rank': by_real.index(chosen) + 1,
            'chosen_gain_over_control': real[chosen] - control_accuracy,
            'regret': real[best] - real[chosen],
            'real_best_rank_by_score': by_mean.index(best) + 1,
            'real_top_in_score_top': len(set(by_real[:top]) & set(by_mean[:top])),
            'mean_measurement_range': sum(spreads) / len(spreads),
            'seconds_per_measurement': sum(row['seconds'] for row in rows) / sum(len(row['scores']) for row in rows)}


def run_scoring_calibration(output, measurements=CALIBRATION_MEASUREMENTS, settings=CALIBRATION_SETTINGS):
    """Phase 15a: score Phase 13's arms with longer scoring and larger simulation sets, against real training."""
    generation = DECISION_REPLAY_GENERATION
    replay = output / 'decision_replay' / f'gen_{generation}'
    replay_summary = json.loads((replay / 'summary.json').read_text(encoding='utf-8'))
    target = output / 'scoring_calibration' / f'gen_{generation}'
    digest = source_fingerprint()
    metadata = {'source_sha256': digest, 'decision_replay': Path(os.path.relpath(replay, output)).as_posix(),
                'settings': [list(setting) for setting in settings], 'measurements': measurements,
                'score_last_epochs': CALIBRATION_SCORE_LAST_EPOCHS, 'search_seed': CALIBRATION_SEARCH_SEED,
                'simulation_set': 'ccs'}
    manifest = target / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Scoring-calibration settings or source changed; use a new output directory')
    else:
        target.mkdir(parents=True, exist_ok=True)
        write_source_snapshot(target / 'source_snapshot.zip', digest)
        write_json(manifest, metadata)
    exp010.apply_values(full_growth_settings(sim_fixes_spec()))
    searches = [json.loads((DECISION_REPLAY_RUN / f'board/simulations/simulation_gen_{index}.json')
                           .read_text(encoding='utf-8')) for index in range(generation + 1)]
    seed = json.loads((DECISION_REPLAY_RUN / 'run.json').read_text(encoding='utf-8'))['seed']
    payload = torch.load(replay / 'decision_model.pt', map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traced = decision_architecture(searches, generation, decision_replay_config(device, payload['lr_scheduler']))
    traced.gm.load_state_dict(payload['state_dict'])
    available = generate_all_actions(traced, decision_replay_config(device, payload['lr_scheduler']))
    arms = [(result, None if result['arm'] == 'control' else recorded_action(available, result['action']))
            for result in replay_summary['results']]
    _, val, clean = get_datasets('cifar10').loaders(128)
    loaders = {}
    for epochs, percent in settings:
        name = calibration_setting_name(epochs, percent)
        for result, action in arms:
            path = target / name / f"{result['arm']}.json"
            if path.exists():
                print(f"SKIP {name} {result['arm']}", flush=True)
                continue
            if percent not in loaders:
                size = round(len(clean.dataset) * percent / 100)
                loaders[percent] = GRID_SIMULATION_SETS['ccs']().sample_loaders(clean, val, size, seed=seed)
            config = decision_replay_config(device, payload['lr_scheduler'])
            config.simulation_scheduler = AlwaysSimulationScheduler(0.0, epochs)
            config.simulation_score = SimulationScore(weight_acc=1.0, weight_countW=0.0,
                                                      average_last_epochs=CALIBRATION_SCORE_LAST_EPOCHS)
            config.set_simulation_loaders(*loaders[percent])
            candidate = copy.deepcopy(traced)
            if action is not None:
                copy.deepcopy(action).execute(candidate)
            print(f"START {name} {result['arm']}", flush=True)
            started = time.monotonic()
            scores = []
            for measurement in range(measurements):
                with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
                    torch.manual_seed(CALIBRATION_SEARCH_SEED + measurement)
                    score, _ = config.simulation_score.score_and_train(candidate.gm, config)
                scores.append(float(score))
            write_json(path, {'arm': result['arm'], 'action': result['action'], 'scores': scores,
                              'real_validation_accuracy': result['final_validation_accuracy'],
                              'phase12_simulation_score': result['simulation_score'],
                              'seconds': time.monotonic() - started,
                              'simulation_train_size': len(loaders[percent][0].dataset),
                              'simulation_validation_size': len(loaders[percent][1].dataset)})
            print(f"DONE {name} {result['arm']}: {', '.join(f'{score:.4f}' for score in scores)}", flush=True)
    control = replay_summary['results'][0]['final_validation_accuracy']
    rows = []
    for epochs, percent in settings:
        name = calibration_setting_name(epochs, percent)
        results = [json.loads((target / name / f"{result['arm']}.json").read_text(encoding='utf-8'))
                   for result, _ in arms]
        rows.append({'setting': name, 'scoring_epochs': epochs, 'simulation_set_percent': percent,
                     'metrics': calibration_metrics(results[1:], control), 'results': results})
    write_json(target / 'summary.json', {'manifest': metadata, 'control_validation_accuracy': control,
                                         'settings': rows})
    if output == OUTPUT:
        from documentation.website.scripts.generate_experiment_015_report import main as generate_report
        generate_report()
    print('COMPLETE', flush=True)


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
                               epochs_values=SCORING_EPOCHS, score_last_epochs=1):
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
    if score_last_epochs != 1:
        metadata['score_last_epochs'] = score_last_epochs
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
                simulation_score=SimulationScore(weight_acc=1, weight_countW=0,
                                                 average_last_epochs=score_last_epochs),
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
    # Monte Carlo 2 now also shuffles expansion by action type, picks random follow-ups by type
    # and uses one training seed per search; this phase adds the time to use them.
    'check-noise-fix': {'label': 'Phase 9', 'folder': 'noise_fix_check', 'scoring_epochs': SCORE_FIX_SCORING_EPOCHS,
                        'simulation_time': 300.0, 'score_last_epochs': 2},
}


def run_score_fix_check(traced, actions, payload, device, seeds, output, seconds, phase='check-score-fix'):
    """Phases 6 and 8: rerun the Phase 3 search on the A08 decision with the score-fix settings."""
    spec = CHECK_PHASES[phase]
    label, scoring_epochs, target = spec['label'], spec['scoring_epochs'], output / spec['folder']
    seconds = spec.get('simulation_time', seconds)
    score_last_epochs = spec.get('score_last_epochs', 1)
    previous = {name: getattr(project_config, name) for name in SCORE_FIX_MCTS_VALUES}
    for name, value in SCORE_FIX_MCTS_VALUES.items():
        setattr(project_config, name, value)
    try:
        for seed in seeds:
            print(f'START {label} check seed={seed}', flush=True)
            run_montecarlo2_comparison(traced, actions, payload, device, seed, target / f'seed_{seed}',
                                       seconds, epochs_values=(scoring_epochs,),
                                       score_last_epochs=score_last_epochs)
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
                                         'scoring_epochs': scoring_epochs, 'score_last_epochs': score_last_epochs,
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


def recorded_action(available, description):
    """The generated action matching a recorded description, with the recorded widening seed restored."""
    matches = [action for action in available if identity(str(action)) == identity(description)]
    if len(matches) != 1:
        raise ValueError(f'Expected one exact match for {description}, found {len(matches)}')
    action = copy.deepcopy(matches[0])
    if isinstance(action, WidenConv):
        match = re.search(r"Widen Conv Action: \['[^']+',\s*\d+,\s*(\d+)\]", description)
        if not match:
            raise ValueError('Recorded widening seed missing')
        action.params[2] = int(match.group(1))
    if str(action) != description:
        raise ValueError(f'Reconstructed action differs from recorded action: {action}')
    return action


def recorded_actions(traced, config, board):
    """Match recorded descriptions and restore the recorded widening seeds."""
    available = generate_all_actions(traced, config)
    result = [(recorded_action(available, row['action']), row) for row in board['candidates']]
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
                                            *FULL_GROWTH_PHASES, 'full-grid', 'full-sim-fixes', 'decision-replay', 'full-widening', 'full-normalization', 'full-residual-stem',
                                            'scoring-calibration',
                                            *CHECK_PHASES),
                        default='replay')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--simulation-time', type=float, default=32.0)
    parser.add_argument('--ucb-c', type=float, nargs='+', choices=GRID_UCB_C, default=list(GRID_UCB_C),
                        help='full-grid only: UCB C values to run')
    parser.add_argument('--simulation-set', nargs='+', choices=tuple(GRID_SIMULATION_SETS),
                        default=list(GRID_SIMULATION_SETS), help='full-grid only: selectors to run')
    args = parser.parse_args()
    if args.epochs < 1 or args.simulation_time < 0:
        parser.error('--epochs must be positive and --simulation-time nonnegative')
    if args.phase == 'decision-replay':
        run_decision_replay(args.output, args.epochs)
        return
    if args.phase == 'scoring-calibration':
        run_scoring_calibration(args.output)
        return
    if args.phase == 'full-widening':
        run_full_growth(args.output, args.seeds, spec=widening_spec())
        return
    if args.phase == 'full-residual-stem':
        run_full_growth(args.output, args.seeds, spec=residual_stem_spec())
        return
    if args.phase == 'full-normalization':
        run_full_growth(args.output, args.seeds, spec=normalization_spec())
        run_full_growth(args.output, args.seeds, spec=normalized_stem_spec())
        return
    if args.phase == 'full-sim-fixes':
        run_full_growth(args.output, args.seeds, spec=sim_fixes_spec())
        return
    if args.phase == 'full-grid':
        for ucb_c in args.ucb_c:
            for simulation_set in args.simulation_set:
                run_full_growth(args.output, args.seeds, spec=grid_spec(ucb_c, simulation_set))
        return
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
            from documentation.website.scripts.generate_experiment_015_report import main as generate_report
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
