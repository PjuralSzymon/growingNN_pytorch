from types import SimpleNamespace

from experiments import train_cifar10_exp014_action_replay as exp014


def test_full_growth_uses_exp010_recipe_and_requested_overrides(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate', lambda *args, **kwargs: calls.append((args, kwargs)))

    exp014.run_full_growth(tmp_path, [1, 2])

    assert len(calls) == 2
    for (args, kwargs), seed in zip(calls, [1, 2]):
        assert args[0:2] == ('cifar10', seed)
        assert args[4]['simulation_time'] == 300.0
        assert args[4]['simulation_set_percent'] == 10.0
        assert args[4]['MCTS_ROLLOUT_DEPTH'] == 1
        assert args[4]['MCTS_PROPAGATE_ROLLOUT_VALUE'] is True
        assert kwargs['generations'] == 10
        assert kwargs['scoring_epochs'] == 20


def test_score_fix_phase_writes_separate_output_with_same_recipe(tmp_path, monkeypatch):
    import json
    import sys

    calls = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'full-score-fix', '--output', str(tmp_path),
                                      '--seeds', '1', '2'])

    exp014.main()

    target = tmp_path / 'full_growth_5min_score_fix'
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['note'] == exp014.SCORE_FIX_NOTE
    assert manifest['algorithm'] == 'montecarlo_alg_2'
    assert manifest['widen_conv'] is True
    assert manifest['scoring_epochs'] == 5
    assert not (tmp_path / 'full_growth_5min').exists()
    assert [args[6] for args, _ in calls] == [target / 'seed_1', target / 'seed_2']
    for args, kwargs in calls:
        assert args[4]['simulation_time'] == 300.0
        assert args[4]['MCTS_ROLLOUT_DEPTH'] == 0
        assert args[4]['MCTS_PROPAGATE_ROLLOUT_VALUE'] is True
        assert kwargs['phase_name'] == 'phase7'
        assert kwargs['experiment_label'] == '014_full_growth_score_fix'
        assert kwargs['experiment_id_prefix'] == 'exp014_score_fix'
        assert kwargs['scoring_epochs'] == 5


def test_widening_search_enables_widen_conv(monkeypatch):
    seen = []
    monkeypatch.setattr(exp014.montecarlo_alg_2, 'get_action',
                        lambda traced, config: seen.append(config.ACTIONS_ENABLE_WIDEN_CONV) or (None, 0, 0))
    config = SimpleNamespace()
    exp014.WideningMonteCarlo2.get_action(object(), config)
    assert seen == [True]


def test_score_fix_check_uses_settings_and_restores_config(tmp_path, monkeypatch):
    import json

    calls = []
    original = {name: getattr(exp014.project_config, name) for name in exp014.SCORE_FIX_MCTS_VALUES}

    def fake_comparison(traced, actions, payload, device, seed, output, seconds, epochs_values, **kwargs):
        calls.append({name: getattr(exp014.project_config, name) for name in exp014.SCORE_FIX_MCTS_VALUES})
        assert epochs_values == (5,)
        exp014.write_json(output / 'montecarlo2_epochs' / 'epochs_5.json', {
            'selected_action': 'A08', 'results': [{'arm': 'candidate_5', 'score': 0.5}]})

    monkeypatch.setattr(exp014, 'run_montecarlo2_comparison', fake_comparison)
    exp014.run_score_fix_check(None, [], {}, 'cpu', [1, 2], tmp_path, 32.0)

    assert calls == [exp014.SCORE_FIX_MCTS_VALUES] * 2
    assert {name: getattr(exp014.project_config, name) for name in original} == original
    summary = json.loads((tmp_path / 'score_fix_check' / 'summary.json').read_text(encoding='utf-8'))
    assert [run['seed'] for run in summary['runs']] == [1, 2]


def test_score_fix_check_with_20_epochs_writes_phase8_folder(tmp_path, monkeypatch):
    seen = []

    def fake_comparison(traced, actions, payload, device, seed, output, seconds, epochs_values, **kwargs):
        seen.append((output, epochs_values))
        exp014.write_json(output / 'montecarlo2_epochs' / 'epochs_20.json', {
            'selected_action': 'A08', 'results': [{'arm': 'candidate_5', 'score': 0.5}]})

    monkeypatch.setattr(exp014, 'run_montecarlo2_comparison', fake_comparison)
    exp014.run_score_fix_check(None, [], {}, 'cpu', [1], tmp_path, 32.0, 'check-score-fix-20')

    assert seen == [(tmp_path / 'score_fix_check_20ep' / 'seed_1', (20,))]
    assert (tmp_path / 'score_fix_check_20ep' / 'summary.json').exists()
    assert not (tmp_path / 'score_fix_check').exists()


def test_noise_fix_check_uses_five_minutes_and_two_epoch_score(tmp_path, monkeypatch):
    seen = []

    def fake_comparison(traced, actions, payload, device, seed, output, seconds, epochs_values,
                        score_last_epochs=1):
        seen.append((output, seconds, epochs_values, score_last_epochs))
        exp014.write_json(output / 'montecarlo2_epochs' / 'epochs_5.json', {
            'selected_action': 'A08', 'results': [{'arm': 'candidate_5', 'score': 0.5}]})

    monkeypatch.setattr(exp014, 'run_montecarlo2_comparison', fake_comparison)
    exp014.run_score_fix_check(None, [], {}, 'cpu', [1], tmp_path, 32.0, 'check-noise-fix')

    assert seen == [(tmp_path / 'noise_fix_check' / 'seed_1', 300.0, (5,), 2)]


def test_grid_cell_runs_with_its_ucb_c_and_simulation_set(tmp_path, monkeypatch):
    import json

    seen = []
    original_sampler, original_search = exp014.exp010.sample_loaders, exp014.exp010.montecarlo_alg
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))

    def fake_run(*args, **kwargs):
        seen.append((args, kwargs, exp014.exp010.sample_loaders, exp014.exp010.montecarlo_alg))

    monkeypatch.setattr(exp014.exp010, 'run_candidate', fake_run)
    exp014.run_full_growth(tmp_path, [1], spec=exp014.grid_spec(0.05, 'ccs'))

    (args, kwargs, sampler, search), = seen
    assert args[4]['MCTS_UCB1_C'] == 0.05 and args[4]['MCTS_ROLLOUT_DEPTH'] == 0
    assert args[6] == tmp_path / 'full_grid' / 'c0p05_ccs' / 'seed_1'
    assert kwargs['phase_name'] == 'phase10' and kwargs['scoring_epochs'] == 5
    assert isinstance(sampler.__self__, exp014.CcsSimulationSet)
    assert isinstance(search, exp014.GridMonteCarlo2)
    manifest = json.loads((tmp_path / 'full_grid' / 'c0p05_ccs' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['simulation_set'] == 'ccs' and manifest['search_budget']['seconds_per_candidate'] == 24.0
    assert exp014.exp010.sample_loaders is original_sampler
    assert exp014.exp010.montecarlo_alg is original_search


def test_grid_search_budget_scales_with_candidates(monkeypatch):
    calls = []
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, config: list(range(40)))
    monkeypatch.setattr(exp014.montecarlo_alg_2, 'get_action',
                        lambda traced, config: calls.append(config) or (None, 0, 0))
    search = exp014.GridMonteCarlo2(24.0, 300.0, score_last_epochs=2)
    config = SimpleNamespace(simulation_score=SimpleNamespace(),
                             simulation_scheduler=SimpleNamespace(simulation_time=300.0))

    search.get_action(object(), config)
    assert config.simulation_scheduler.simulation_time == 960.0
    assert config.ACTIONS_ENABLE_WIDEN_CONV is True and config.simulation_score.average_last_epochs == 2

    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, config: list(range(5)))
    search.get_action(object(), config)
    assert config.simulation_scheduler.simulation_time == 300.0


def test_sim_fixes_phase_runs_the_best_phase10_cell_in_its_own_folder(tmp_path, monkeypatch):
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args, kwargs, exp014.exp010.sample_loaders)))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'full-sim-fixes', '--output', str(tmp_path),
                                      '--seeds', '1', '2'])

    exp014.main()

    assert [args[6] for args, _, _ in seen] == [tmp_path / 'full_growth_sim_fixes' / f'seed_{seed}' for seed in (1, 2)]
    for args, kwargs, sampler in seen:
        assert args[4]['MCTS_UCB1_C'] == 1.0 and args[4]['MCTS_ROLLOUT_DEPTH'] == 0
        assert kwargs['phase_name'] == 'phase12' and kwargs['scoring_epochs'] == 5
        assert isinstance(sampler.__self__, exp014.CcsSimulationSet)



def test_widening_phase_repeats_phase12_in_its_own_folder(tmp_path, monkeypatch):
    import json
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args, kwargs, exp014.exp010.sample_loaders)))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'full-widening', '--output', str(tmp_path),
                                      '--seeds', '1', '2', '3'])

    exp014.main()

    target = tmp_path / 'full_growth_widening'
    assert [args[6] for args, _, _ in seen] == [target / f'seed_{seed}' for seed in (1, 2, 3)]
    for args, kwargs, sampler in seen:
        assert args[4]['MCTS_UCB1_C'] == 1.0 and args[4]['MCTS_ROLLOUT_DEPTH'] == 0
        assert kwargs['phase_name'] == 'phase14' and kwargs['scoring_epochs'] == 5
        assert isinstance(sampler.__self__, exp014.CcsSimulationSet)
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['widen_conv'] is True and manifest['note'] == exp014.WIDENING_NOTE

class _RecordedAction:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def execute(self, traced):
        pass


class _ShrinkingAction(_RecordedAction):
    pass


def test_recorded_choices_return_the_selected_actions_in_order(monkeypatch):
    monkeypatch.setattr(exp014, 'generate_all_actions',
                        lambda traced, config: [_RecordedAction('a'), _RecordedAction('b')])
    choices = exp014.RecordedChoices(['b', 'a'])

    assert [str(choices.get_action(None, None)[0]) for _ in range(2)] == ['b', 'a']
    assert choices.get_action(None, None) == (None, None, None)


def test_decision_replay_trains_control_and_growing_candidates_from_one_model(tmp_path, monkeypatch):
    import json

    import torch

    traced = exp014.TracedModel.create(exp014.exp010.initial_graph('cifar10'), (1, 3, 32, 32))
    params = traced.param_count()
    candidates = [{'action': name, 'score': score, 'visits': 2, 'chosen': name == 'grow', 'paramCount': params}
                  for name, score in (('grow', 0.5), ('shrink', 0.6), ('widen', 0.4))]
    run = tmp_path / 'run'
    (run / 'board/simulations').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'seed': 2}), encoding='utf-8')
    for generation in range(4):
        (run / f'board/simulations/simulation_gen_{generation}.json').write_text(json.dumps(
            {'actionChosen': 'grow', 'paramCountBefore': params, 'candidates': candidates}), encoding='utf-8')
    rebuilt, trained = [], []

    def fake_rebuild(searches, generation, seed, data, device):
        rebuilt.append((len(searches), generation, seed))
        return traced.gm.state_dict(), exp014.CosineDecayScheduler(200), 0.5

    def fake_training(model, epochs, *args, **kwargs):
        trained.append(epochs)
        return model, {'val_acc': [0.1 * len(trained)]}

    monkeypatch.setattr(exp014, 'DECISION_REPLAY_RUN', run)
    monkeypatch.setattr(exp014, 'DECISION_REPLAY_SKIPPED', (_ShrinkingAction,))
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014.exp010, 'apply_values', lambda values: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(loaders=lambda size: (None, None, None)))
    monkeypatch.setattr(exp014, 'with_cutout', lambda loader: loader)
    monkeypatch.setattr(exp014, 'train_decision_model', fake_rebuild)
    monkeypatch.setattr(exp014, 'decision_architecture', lambda searches, generation, config: traced)
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, config: [
        _RecordedAction('grow'), _ShrinkingAction('shrink'), _RecordedAction('widen')])
    monkeypatch.setattr(exp014, 'gradient_descent', fake_training)

    exp014.run_decision_replay(tmp_path, epochs=20)

    target = tmp_path / 'decision_replay/gen_3'
    summary = json.loads((target / 'summary.json').read_text(encoding='utf-8'))
    assert rebuilt == [(4, 3, 2)]
    assert [result['action'] for result in summary['results']] == ['unchanged', 'grow', 'widen']
    assert [result['chosen'] for result in summary['results']] == [False, True, False]
    assert trained == [20, 20, 20]
    assert torch.load(target / 'decision_model.pt', weights_only=False)['validation_accuracy'] == 0.5

    trained.clear()
    exp014.run_decision_replay(tmp_path, epochs=20)
    assert rebuilt == [(4, 3, 2)] and trained == []


def test_decision_architecture_applies_the_recorded_choices_before_the_generation(monkeypatch):
    applied = []

    class Logged(_RecordedAction):
        def execute(self, traced):
            applied.append(self.name)

    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, config: [Logged('a'), Logged('b')])
    searches = [{'actionChosen': name} for name in ('b', 'a', 'b')]

    exp014.decision_architecture(searches, 2, None)

    assert applied == ['b', 'a']


def _calibration_row(arm, scores, real):
    return {'arm': arm, 'scores': scores, 'real_validation_accuracy': real, 'seconds': 3.0 * len(scores)}


def test_calibration_metrics_compare_scores_with_real_training():
    rows = [_calibration_row('a', [0.50, 0.40], 0.55), _calibration_row('b', [0.30, 0.60], 0.60),
            _calibration_row('c', [0.20, 0.20], 0.50)]

    metrics = exp014.calibration_metrics(rows, control_accuracy=0.52, top=1)

    assert metrics['spearman_one_measurement'] == 0.5
    assert metrics['spearman_mean'] == 0.5
    assert metrics['chosen'] == 'a' and metrics['chosen_real_rank'] == 2
    assert round(metrics['chosen_gain_over_control'], 6) == 0.03 and round(metrics['regret'], 6) == 0.05
    assert metrics['real_best_rank_by_score'] == 2 and metrics['real_top_in_score_top'] == 0
    assert round(metrics['mean_measurement_range'], 6) == 0.133333 and metrics['seconds_per_measurement'] == 3.0


def test_scoring_calibration_scores_every_arm_per_setting_and_resumes(tmp_path, monkeypatch):
    import json

    import torch

    traced = exp014.TracedModel.create(exp014.exp010.initial_graph('cifar10'), (1, 3, 32, 32))
    replay = tmp_path / 'decision_replay/gen_3'
    replay.mkdir(parents=True)
    torch.save({'state_dict': traced.gm.state_dict(), 'lr_scheduler': exp014.CosineDecayScheduler(200)},
               replay / 'decision_model.pt')
    results = [{'arm': 'control', 'action': 'unchanged', 'final_validation_accuracy': 0.5, 'simulation_score': None}]
    results += [{'arm': f'candidate_0{index}', 'action': name, 'final_validation_accuracy': real,
                 'simulation_score': 0.4} for index, (name, real) in enumerate((('grow', 0.6), ('widen', 0.55)), 1)]
    (replay / 'summary.json').write_text(json.dumps({'results': results}), encoding='utf-8')
    run = tmp_path / 'run'
    (run / 'board/simulations').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'seed': 2}), encoding='utf-8')
    for generation in range(4):
        (run / f'board/simulations/simulation_gen_{generation}.json').write_text('{}', encoding='utf-8')
    sampled, scored = [], []

    class FakeSet:
        def sample_loaders(self, clean, val, size, seed):
            sampled.append((size, seed))
            return SimpleNamespace(dataset=range(size)), SimpleNamespace(dataset=range(size // 4))

    class FakeScore:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def score_and_train(self, model, config):
            scored.append((config.simulation_scheduler.simulation_epochs, len(config.sim_train_loader.dataset),
                           self.kwargs['average_last_epochs']))
            return 0.1 * len(scored), model

    clean = SimpleNamespace(dataset=range(1000))
    monkeypatch.setattr(exp014, 'DECISION_REPLAY_RUN', run)
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014.exp010, 'apply_values', lambda values: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(loaders=lambda size: (None, None, clean)))
    monkeypatch.setattr(exp014, 'decision_architecture', lambda searches, generation, config: traced)
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, config: [
        _RecordedAction('grow'), _RecordedAction('widen')])
    monkeypatch.setattr(exp014, 'recorded_action', lambda available, description: next(
        action for action in available if str(action) == description))
    monkeypatch.setattr(exp014, 'GRID_SIMULATION_SETS', {'ccs': FakeSet})
    monkeypatch.setattr(exp014, 'SimulationScore', FakeScore)
    monkeypatch.setattr(exp014, 'calibration_metrics', lambda rows, control: {'arms': [row['arm'] for row in rows],
                                                                              'control': control})

    exp014.run_scoring_calibration(tmp_path, measurements=2, settings=((5, 10.0), (10, 25.0)))

    target = tmp_path / 'scoring_calibration/gen_3'
    summary = json.loads((target / 'summary.json').read_text(encoding='utf-8'))
    assert sampled == [(100, 2), (250, 2)]
    assert scored == [(5, 100, 2)] * 6 + [(10, 250, 2)] * 6
    assert [row['setting'] for row in summary['settings']] == ['epochs_5_set_10', 'epochs_10_set_25']
    assert summary['settings'][0]['metrics'] == {'arms': ['candidate_01', 'candidate_02'], 'control': 0.5}
    first = json.loads((target / 'epochs_5_set_10/candidate_01.json').read_text(encoding='utf-8'))
    assert len(first['scores']) == 2 and first['real_validation_accuracy'] == 0.6

    scored.clear()
    exp014.run_scoring_calibration(tmp_path, measurements=2, settings=((5, 10.0), (10, 25.0)))
    assert scored == [] and sampled == [(100, 2), (250, 2)]


def test_normalization_phase_is_a_short_widening_run_with_normalization_candidates(tmp_path, monkeypatch):
    import json
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args, kwargs, exp014.exp010.montecarlo_alg,
                                                             exp014.exp010.initial_graph('cifar10'))))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'full-normalization', '--output', str(tmp_path),
                                      '--seeds', '1'])
    original_starter = exp014.exp010.initial_graph

    exp014.main()

    target = tmp_path / 'full_growth_normalization'
    (args, kwargs, search, starter), (stem_args, stem_kwargs, _, stem_starter) = seen
    assert type(starter).__name__ == 'TinyCifar10Starter'
    assert isinstance(stem_starter, exp014.NormalizedStemStarter)
    assert exp014.exp010.initial_graph is original_starter
    assert stem_args[6] == tmp_path / 'full_growth_normalized_stem' / 'seed_1'
    assert stem_kwargs['phase_name'] == 'phase16b' and stem_kwargs['generations'] == exp014.NORMALIZATION_GENERATIONS
    stem_manifest = json.loads((tmp_path / 'full_growth_normalized_stem' / 'manifest.json').read_text(encoding='utf-8'))
    assert stem_manifest['starter_name'] == 'TinyCifar10Starter + BatchNorm after conv1'
    assert args[6] == target / 'seed_1'
    assert kwargs['phase_name'] == 'phase16' and kwargs['generations'] == exp014.NORMALIZATION_GENERATIONS
    assert kwargs['scoring_epochs'] == 5 and args[4]['MCTS_UCB1_C'] == 1.0
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['generations'] == exp014.NORMALIZATION_GENERATIONS and manifest['add_normalization'] is True

    config = SimpleNamespace(ACTIONS_ENABLE_ADD_NORMALIZATION=False, simulation_score=SimpleNamespace(),
                             simulation_scheduler=SimpleNamespace())
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, cfg: [1, 2])
    monkeypatch.setattr(exp014.montecarlo_alg_2, 'get_action', lambda traced, cfg: 'chosen')
    assert search.get_action(None, config) == 'chosen'
    assert config.ACTIONS_ENABLE_ADD_NORMALIZATION is True and config.ACTIONS_ENABLE_WIDEN_CONV is True


def test_normalization_candidates_get_their_own_group_in_reports():
    from documentation.website.scripts.generate_experiment_017_action_scores import group_key
    assert group_key({'action': " ( Add Normalization Action: ['conv1', 'BatchNorm2d', 8, 'norm_0'] ) "}) \
        == 'Add normalization'


def test_normalized_stem_starter_is_the_tiny_starter_with_batch_norm_before_relu():
    import torch
    from growingnn.core.traced_model import TracedModel
    traced = TracedModel.create(exp014.NormalizedStemStarter(), (1, 3, 32, 32))
    calls = [node for node in traced.gm.graph.nodes if node.op in ('call_module', 'call_function')]
    names = [str(node.target) for node in calls]
    assert names[:3] == ['conv1', 'conv1_bn', str(torch.relu)]
    assert traced.gm.conv1.out_channels == 8 and traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_residual_stem_phase_starts_from_the_residual_block_starter(tmp_path, monkeypatch):
    import json
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args, kwargs, exp014.exp010.initial_graph('cifar10'))))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'full-residual-stem', '--output', str(tmp_path),
                                      '--seeds', '1'])

    exp014.main()

    (args, kwargs, starter), = seen
    assert isinstance(starter, exp014.ResidualStemStarter)
    assert args[6] == tmp_path / 'full_growth_residual_stem' / 'seed_1'
    assert kwargs['phase_name'] == 'phase16c' and kwargs['generations'] == exp014.NORMALIZATION_GENERATIONS
    manifest = json.loads((tmp_path / 'full_growth_residual_stem' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['add_normalization'] is True and 'residual block' in manifest['starter_name']


def test_residual_stem_starter_adds_the_block_output_to_the_stem():
    import torch
    from growingnn.core.traced_model import TracedModel
    from growingnn.utils.fx.sum_nodes import is_sum_node
    traced = TracedModel.create(exp014.ResidualStemStarter(), (1, 3, 32, 32))
    sums = [node for node in traced.gm.graph.nodes if is_sum_node(node)]
    assert len(sums) == 1
    assert {str(arg.target) for arg in sums[0].all_input_nodes} == {str(torch.relu), 'block_bn'}
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_long_run_phase_grows_the_residual_starter_for_fifteen_generations(tmp_path, monkeypatch):
    import json
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args, kwargs, exp014.exp010.montecarlo_alg,
                                                             exp014.exp010.initial_graph('cifar10'))))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'long-run', '--output', str(tmp_path), '--seeds', '1'])

    exp014.main()

    (args, kwargs, search, starter), = seen
    assert isinstance(starter, exp014.ResidualStemStarter)
    assert args[6] == tmp_path / 'full_growth_long' / 'seed_1'
    assert kwargs['phase_name'] == 'phase17' and kwargs['generations'] == 15 and kwargs['scoring_epochs'] == 10
    manifest = json.loads((tmp_path / 'full_growth_long' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['search_budget']['maximum_seconds'] == 1500 and manifest['generations'] == 15


def test_long_run_search_uses_growth_actions_caps_its_time_and_stops_after_the_deadline(monkeypatch):
    from growingnn.core.config import RunningConfig
    search = exp014.LongRunMonteCarlo2(60.0, 300.0, 1500.0, search_hours=1.0, score_last_epochs=2)
    config = RunningConfig(generations=1, epochs=1)
    config.simulation_score = SimpleNamespace()
    config.simulation_scheduler = SimpleNamespace()
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, cfg: list(range(40)))
    monkeypatch.setattr(exp014.montecarlo_alg_2, 'get_action', lambda traced, cfg: ('chosen', 0, 0))

    assert search.get_action(None, config) == ('chosen', 0, 0)
    assert config.simulation_scheduler.simulation_time == 1500.0
    assert not config.ACTIONS_ENABLE_DEL_LAYER and not config.ACTIONS_ENABLE_RESIDUAL_BLOCK_SHRINK
    assert not config.ACTIONS_ENABLE_ADD_SEQ_DROPOUT_05 and config.ACTIONS_ENABLE_ADD_NORMALIZATION

    search.deadline -= 7200
    assert search.get_action(None, config) == (None, 0, 0)


def test_long_run_baselines_run_random_growth_then_no_growth_for_each_seed(tmp_path, monkeypatch):
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args[6], kwargs, exp014.exp010.montecarlo_alg)))
    monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', 'long-run-baselines', '--output', str(tmp_path),
                                      '--seeds', '1', '2'])

    exp014.main()

    assert [path.relative_to(tmp_path).as_posix() for path, _, _ in seen] == [
        'full_growth_long_random/seed_1', 'full_growth_long_random/seed_2',
        'full_growth_long_static/seed_1', 'full_growth_long_static/seed_2']
    assert all(kwargs['generations'] == 15 for _, kwargs, _ in seen)
    assert isinstance(seen[0][2], exp014.RandomGrowth) and seen[2][2] is exp014.NoGrowth


def test_random_growth_picks_an_executable_growth_action_and_no_growth_picks_none():
    import random
    import torch
    from growingnn.core.config import RunningConfig
    from growingnn.core.traced_model import TracedModel
    random.seed(0)
    traced = TracedModel.create(exp014.ResidualStemStarter(), (1, 3, 32, 32))
    config = RunningConfig(generations=1, epochs=1)
    search = exp014.RandomGrowth(0.0, 0.0, 0.0, search_hours=1.0, score_last_epochs=2)
    for _ in range(5):
        action, _, _ = search.get_action(traced, config)
        assert type(action).__name__ not in ('AddSeqDropoutLayer', 'DelLayer', 'DelNeurons')
        action.execute(traced)
    assert traced.gm(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert exp014.NoGrowth.get_action(traced, config) == (None, 0, 0)


def test_comparison_phases_run_each_monte_carlo_version_with_30_epochs_per_generation(tmp_path, monkeypatch):
    import json
    import sys

    seen = []
    monkeypatch.setattr(exp014, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(exp014, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(exp014, 'get_datasets', lambda name: SimpleNamespace(name=name))
    monkeypatch.setattr(exp014.exp010, 'run_candidate',
                        lambda *args, **kwargs: seen.append((args[6], kwargs, exp014.exp010.montecarlo_alg,
                                                             exp014.exp010.EPOCHS)))
    for phase in ('mc1', 'mc2'):
        monkeypatch.setattr(sys, 'argv', ['exp014', '--phase', phase, '--output', str(tmp_path), '--seeds', '1', '2'])
        exp014.main()

    assert [path.relative_to(tmp_path).as_posix() for path, _, _, _ in seen] == [
        'mc_comparison_mc1/seed_1', 'mc_comparison_mc1/seed_2', 'mc_comparison_mc2/seed_1', 'mc_comparison_mc2/seed_2']
    assert all(kwargs['generations'] == 15 and epochs == 30 for _, kwargs, _, epochs in seen)
    assert seen[0][2].algorithm is exp014.montecarlo_alg and seen[2][2].algorithm is exp014.montecarlo_alg_2
    assert all(search.search_hours is None for _, _, search, _ in seen)
    assert exp014.exp010.EPOCHS == 20
    manifest = json.loads((tmp_path / 'mc_comparison_mc1' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['algorithm'] == 'montecarlo_alg' and manifest['epochs_per_generation'] == 30


def test_search_without_deadline_searches_every_generation(monkeypatch):
    from growingnn.core.config import RunningConfig
    search = exp014.LongRunMonteCarlo2(60.0, 300.0, 1500.0, None, score_last_epochs=2,
                                       algorithm=exp014.montecarlo_alg)
    config = RunningConfig(generations=1, epochs=1)
    config.simulation_score = SimpleNamespace()
    config.simulation_scheduler = SimpleNamespace()
    monkeypatch.setattr(exp014, 'generate_all_actions', lambda traced, cfg: list(range(3)))
    monkeypatch.setattr(exp014.montecarlo_alg, 'get_action', lambda traced, cfg: ('chosen', 0, 0))
    monkeypatch.setattr(exp014.time, 'time', lambda: 1e12)

    assert search.get_action(None, config) == ('chosen', 0, 0)
    assert config.simulation_scheduler.simulation_time == 300.0
