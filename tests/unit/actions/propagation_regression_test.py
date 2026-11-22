"""Width-propagation cases found by running every offered action on varied and randomly grown networks."""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from growingnn.actions.add_neurons import AddNeurons
from growingnn.actions.add_res_conv_layer import AddResConvLayer
from growingnn.actions.add_seq_linear_layer import AddSeqLinearLayer
from growingnn.actions.delete_neurons import DelNeurons
from growingnn.actions.widen_conv import WidenConv
from growingnn.core.traced_model import TracedModel


class LinearToConv(nn.Module):
    """A linear output reshaped into an image: its width is fixed by the view."""

    def __init__(self):
        super().__init__()
        self.inp, self.proj, self.conv, self.head = nn.Linear(6, 12), nn.Linear(12, 16), nn.Conv2d(1, 2, 3, padding=1), nn.Linear(2, 3)

    def forward(self, x):
        x = self.proj(F.relu(self.inp(x))).view(-1, 1, 4, 4)
        return self.head(F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1))


class SequenceFfn(nn.Module):
    """Transformer-style x + W2 relu(W1 LN(x)) on (batch, tokens, features), averaged over tokens."""

    def __init__(self):
        super().__init__()
        self.embed, self.norm, self.up, self.down, self.head = (nn.Linear(5, 8), nn.LayerNorm(8), nn.Linear(8, 16),
                                                                nn.Linear(16, 8), nn.Linear(8, 3))

    def forward(self, x):
        x = self.embed(x)
        x = x + self.down(F.relu(self.up(self.norm(x))))
        return self.head(x.mean(1))


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.mix, self.head = nn.Linear(6, 8), nn.Linear(6, 5), nn.Linear(13, 9), nn.Linear(9, 3)

    def forward(self, x):
        return self.head(F.relu(self.mix(torch.cat([F.relu(self.a(x)), F.relu(self.b(x))], 1))))


class ConvHidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv, self.hidden, self.head = nn.Conv2d(3, 8, 3, padding=1), nn.Linear(8, 16), nn.Linear(16, 10)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(F.relu(self.conv(x)), 1).flatten(1)
        return self.head(F.relu(self.hidden(x)))


def _neuron_actions(traced):
    return ([a for r in (1.1, 1.5, 2.0) for a in AddNeurons.generate_all_actions(traced, ratio=r)]
            + [a for r in (0.1, 0.5, 0.9) for a in DelNeurons.generate_all_actions(traced, ratio=r)])


def _every_offered_neuron_action_works(model, shape):
    traced = TracedModel.create(model, (1, *shape[1:]))
    inputs = torch.randn(*shape)
    expected = traced.gm(inputs).shape
    actions = _neuron_actions(traced)
    for action in actions:
        candidate = copy.deepcopy(traced)
        action.execute(candidate)
        for mode in (True, False):
            candidate.gm.train(mode)
            assert candidate.gm(inputs).shape == expected, str(action)
    return actions


def test_neurons_feeding_a_view_are_not_offered():
    actions = _every_offered_neuron_action_works(LinearToConv(), (4, 6))
    assert all(action.params[0] != 'proj' for action in actions)


def test_neuron_actions_in_a_residual_sequence_model_keep_it_working():
    actions = _every_offered_neuron_action_works(SequenceFfn(), (4, 7, 5))
    assert any(action.params[0] == 'up' for action in actions)


def test_widening_into_a_residual_conv_to_linear_branch_resizes_the_branch_input():
    traced = TracedModel.create(ConvHidden(), (1, 3, 8, 8))
    branch = next(a for a in AddResConvLayer.generate_all_actions(traced) if a.params[:2] == ['conv', 'hidden'])
    branch.execute(traced)
    inputs = torch.randn(2, 3, 8, 8)
    traced.gm.eval()
    before = traced.gm(inputs).detach()
    WidenConv(['conv', 16]).execute(traced)
    assert getattr(traced.gm, branch.params[3])[0].in_channels == 16
    torch.testing.assert_close(traced.gm(inputs), before, rtol=1e-5, atol=1e-6)


def test_sequential_linear_after_a_concatenation_reads_the_concatenated_width():
    traced = TracedModel.create(Concat(), (1, 6))
    inputs = torch.randn(4, 6)
    before = traced.gm(inputs).detach()
    for action in AddSeqLinearLayer.generate_all_actions(traced):
        candidate = copy.deepcopy(traced)
        action.execute(candidate)
        assert candidate.gm(inputs).shape == before.shape, str(action)


class LinearThenLayerNorm(nn.Module):
    """A LayerNorm two layers after the resized one: the second linear absorbs the new width first."""

    def __init__(self):
        super().__init__()
        self.first, self.second, self.norm, self.head = nn.Linear(6, 10), nn.Linear(10, 8), nn.LayerNorm(8), nn.Linear(8, 3)

    def forward(self, x):
        return self.head(self.norm(self.second(F.relu(self.first(x)))))


class TokenMean(nn.Module):
    """A reduction the propagation rules do not know, between the resized layer and the head."""

    def __init__(self):
        super().__init__()
        self.hidden, self.head = nn.Linear(5, 8), nn.Linear(8, 3)

    def forward(self, x):
        return self.head(F.relu(self.hidden(x)).mean(1))


def _verdict(model, layer, shape):
    from growingnn.utils.fx import ModuleResolver, NodeWidthAnalyser
    traced = TracedModel.create(model, shape)
    node = ModuleResolver.find_call_module(traced.gm.graph.nodes, layer)
    return traced, NodeWidthAnalyser.propagation_verdict(traced.gm, node)


def test_propagation_stops_at_the_layer_that_absorbs_the_new_width():
    from growingnn.utils.fx import PropagationVerdict
    assert _verdict(LinearThenLayerNorm(), 'first', (1, 6))[1] is PropagationVerdict.SAFE
    assert _verdict(LinearThenLayerNorm(), 'second', (1, 6))[1] is PropagationVerdict.BLOCKED


def test_a_view_of_the_new_width_blocks_the_resize():
    from growingnn.utils.fx import PropagationVerdict
    assert _verdict(LinearToConv(), 'proj', (1, 6))[1] is PropagationVerdict.BLOCKED


def test_an_unknown_operation_is_resized_only_after_a_trial_on_a_copy():
    from growingnn.actions.utils.layer_resize import can_resize_linear_output
    from growingnn.utils.fx import PropagationVerdict
    traced, verdict = _verdict(TokenMean(), 'hidden', (1, 4, 5))
    assert verdict is PropagationVerdict.UNKNOWN
    assert not can_resize_linear_output(traced.gm, 'hidden', 16)
    assert not can_resize_linear_output(traced.gm, 'hidden', 16, traced.probe())
