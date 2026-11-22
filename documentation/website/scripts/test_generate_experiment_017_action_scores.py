"""Tests for action-score chart labels and board extraction."""

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).with_name('generate_experiment_017_action_scores.py')
    spec = importlib.util.spec_from_file_location('action_score_charts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convolution_flags_and_location_are_visible():
    chart = _module()
    row = {'name': 'Add Seq Conv Layer Action',
           'action': " ( Add Seq Conv Layer Action: ['conv1', 'linear2', Conv2d(64, 64), 'seq_conv_0', True, True, True] ) "}
    assert chart.classify(row) == 'Sequential convolution'
    assert 'conv1 → linear2' in chart.label(row)
    assert 'BN + ReLU' in chart.label(row)
    row['action'] = row['action'].replace('True, True, True', 'True')
    assert 'no BN/ReLU' in chart.label(row)


def test_board_extraction_keeps_candidate_mean_and_visits(tmp_path):
    chart = _module()
    directory = tmp_path / 'simulations'
    directory.mkdir()
    rows = [{'name': 'Add Neurons Action', 'action': " ( Add Neurons Action: ['a', 1.1] ) ",
             'score': 0.42, 'visits': 2, 'chosen': True},
            {'name': 'unknown', 'action': 'unknown', 'score': None}]
    (directory / 'simulation_gen_3.json').write_text(json.dumps({'generation': 3, 'candidates': rows}))
    results = chart.read_searches(tmp_path)
    assert len(results) == 1
    assert results[0][1] == [rows[0]]


def test_family_positions_stay_fixed_when_other_families_are_missing():
    chart = _module()
    widen = {'action': " ( Widen Conv Action: ['conv1', 16, 1] ) ", 'score': 0.4}
    conv = {'action': " ( Add Seq Conv Layer Action: ['conv1', 'linear2', Conv2d(64, 64), 'c', True, True, True] ) ",
            'score': 0.5}
    linear = {'action': " ( Add Seq Linear Layer Action: ['conv1', 'linear2', x, 'l'] ) ", 'score': 0.6}
    searches = [({'generation': 0}, [widen, conv, linear]),
                ({'generation': 1}, [conv])]
    layout = chart.group_layout(searches)
    first = dict((id(row), pos) for pos, row in chart.placed_rows(searches[0][1], layout))
    second = dict((id(row), pos) for pos, row in chart.placed_rows(searches[1][1], layout))
    assert first[id(conv)] == second[id(conv)]
    assert layout['Widen convolution']['positions'][0] < second[id(conv)]
    assert layout['Sequential linear']['positions'][0] > second[id(conv)]


def test_convolution_variants_have_separate_fixed_groups():
    chart = _module()
    conv = {'action': " ( Add Seq Conv Layer Action: ['a', 'b', x, 'c', True, True, True] ) ", 'score': 0.5}
    plain = {'action': " ( Add Seq Conv Layer Action: ['a', 'b', x, 'c', True] ) ", 'score': 0.4}
    layout = chart.group_layout([({'generation': 0}, [conv]), ({'generation': 1}, [plain])])
    assert chart.placed_rows([conv], layout)[0][0] != chart.placed_rows([plain], layout)[0][0]


def test_exact_actions_have_fixed_distinct_positions():
    chart = _module()
    small = {'action': " ( Widen Conv Action: ['conv1', 16, 111] ) ", 'score': 0.4}
    big = {'action': " ( Widen Conv Action: ['conv1', 32, 222] ) ", 'score': 0.5}
    repeated = {'action': " ( Widen Conv Action: ['conv1', 32, 333] ) ", 'score': 0.6}
    layout = chart.group_layout([({'generation': 0}, [small, big]),
                                 ({'generation': 1}, [repeated])])
    first = {chart.action_identity(row): pos for pos, row in chart.placed_rows([small, big], layout)}
    second = {chart.action_identity(row): pos for pos, row in chart.placed_rows([repeated], layout)}
    assert first[chart.action_identity(big)] == second[chart.action_identity(repeated)]
    assert first[chart.action_identity(small)] != first[chart.action_identity(big)]
