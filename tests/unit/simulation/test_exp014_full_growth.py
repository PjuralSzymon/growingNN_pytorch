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
