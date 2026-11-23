import json
from types import SimpleNamespace

import pytest
import torch

from experiments import train_cifar10_exp014_bigger_datasets as exp014


@pytest.mark.parametrize('dataset, size, classes', [('cifar100', 32, 100), ('tiny_imagenet', 64, 200)])
def test_starter_outputs_one_score_per_class(dataset, size, classes):
    starter = exp014.spec(dataset)['starter']()
    assert starter(torch.zeros(2, 3, size, size)).shape == (2, classes)


def test_tiny_imagenet_starter_halves_the_feature_maps():
    starter = exp014.spec('tiny_imagenet')['starter']()
    assert starter.conv1(torch.zeros(1, 3, 64, 64)).shape[-1] == 32


@pytest.mark.parametrize('dataset', ['cifar100', 'tiny_imagenet'])
def test_run_uses_the_dataset_and_its_settings(dataset, tmp_path, monkeypatch):
    calls, loaded = [], []
    base = exp014.exp013
    monkeypatch.setattr(base, 'source_fingerprint', lambda: 'test-fingerprint')
    monkeypatch.setattr(base, 'write_source_snapshot', lambda path, digest: None)
    monkeypatch.setattr(base, 'get_datasets', lambda name: loaded.append(name) or SimpleNamespace(name=name))
    monkeypatch.setattr(base.exp010, 'run_candidate', lambda *args, **kwargs: calls.append((args, kwargs)))

    base.run_full_growth(tmp_path, [1], spec=exp014.spec(dataset))

    values = exp014.DATASETS[dataset]
    assert loaded == [dataset]
    (args, kwargs), = calls
    assert args[0:2] == (dataset, 1)
    assert args[4]['simulation_set_percent'] == values['simulation_percent']
    assert kwargs['generations'] == values['generations']
    assert kwargs['scoring_epochs'] == values['scoring_epochs']
    manifest = json.loads((tmp_path / dataset / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['dataset'] == dataset
    assert manifest['epochs_per_generation'] == values['epochs']
    assert manifest['simulation_set'] == values['simulation_set']
