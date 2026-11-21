"""Widen an ungrouped convolution without changing the network's outputs."""

import copy
import random

import torch
from torch import nn

from growingnn.actions.action import Action
from growingnn.actions.utils.layer_resize import resize_conv_output


class WidenConv(Action):
    """params: [layer name, new channel count, seed for the symmetry-breaking noise]."""

    def _execute(self, traced):
        name, channels = self.params[:2]
        if len(self.params) < 3:
            resize_conv_output(traced.gm, name, channels)
            return
        # A fixed seed makes replaying the same action reproduce the same weights.
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.manual_seed(self.params[2])
            resize_conv_output(traced.gm, name, channels)

    @staticmethod
    def generate_all_actions(traced, ratios=(2, 4), max_channels=256):
        """Offer each widening only if it leaves the evaluation-mode outputs unchanged."""
        actions = []
        for name, module in traced.gm.named_modules():
            if not isinstance(module, nn.Conv2d) or module.groups != 1:
                continue
            for ratio in ratios:
                width = module.out_channels * ratio
                if width > max_channels:
                    continue
                action = WidenConv([name, width, random.randrange(2**32)])
                if WidenConv._preserves_outputs(traced, action):
                    actions.append(action)
        return actions

    @staticmethod
    def _preserves_outputs(traced, action):
        candidate = copy.deepcopy(traced)
        candidate.gm.eval()
        try:
            with torch.no_grad():
                probe = candidate.probe()
                expected = candidate.gm(probe)
                action.execute(candidate)
                actual = candidate.gm(probe)
        except (ValueError, TypeError, RuntimeError):
            return False
        return actual.shape == expected.shape and torch.allclose(actual, expected, rtol=1e-4, atol=1e-5)

    def __str__(self):
        return f" ( Widen Conv Action: {self.params} ) "
