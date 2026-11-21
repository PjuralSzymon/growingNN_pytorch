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
