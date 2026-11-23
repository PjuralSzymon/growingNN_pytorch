"""Widen an ungrouped convolution without changing the network's outputs."""

import random

import torch
from torch import nn

from growingnn.actions.action import Action
from growingnn.actions.utils.layer_resize import can_resize_conv_output, resize_conv_output


class WidenConv(Action):
    """params: [layer name, new channel count, seed for the new channels' random weights]."""

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
        """Offer each widening whose new width the following layers can absorb."""
        actions = []
        for name, module in traced.gm.named_modules():
            if not isinstance(module, nn.Conv2d) or not can_resize_conv_output(traced.gm, name):
                continue
            for ratio in ratios:
                width = module.out_channels * ratio
                if width <= max_channels:
                    actions.append(WidenConv([name, width, random.randrange(2**32)]))
        return actions

    def __str__(self):
        return f" ( Widen Conv Action: {self.params} ) "
