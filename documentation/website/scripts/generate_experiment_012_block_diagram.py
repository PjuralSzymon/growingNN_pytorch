"""Draw the residual blocks the duplicate action finds in a small ResNet, before and after one duplicate.

Uses the project's own graph drawer (draw_filtered_fx_graph). Every block that the action can duplicate
gets its own colour; downsampling blocks, which it skips, are gray. The blocks come from the action itself
(discover_residual_regions), so the figure shows exactly what the search can duplicate or remove.
"""

import sys
from pathlib import Path

import torch.nn as nn
from PIL import Image
from torchvision.models.resnet import BasicBlock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from growingnn.actions.residual_block import ResidualBlockAction, discover_residual_regions  # noqa: E402
from growingnn.core.traced_model import TracedModel  # noqa: E402
from growingnn.utils.fx_graph_drawer import draw_filtered_fx_graph  # noqa: E402

OUTPUT = ROOT / 'documentation/website/app/public/assets/experiments'
PALETTE = ['#aec7e8', '#98df8a', '#ffbb78', '#ff9896', '#c5b0d5']  # light blue, green, orange, red, purple
SKIPPED = '#dddddd'
DUPLICATED_BLOCK = 1  # the second duplicable block (green) is duplicated


class SmallResNet(nn.Module):
    """ResNet-18's building blocks, cut down to fit one figure: two blocks at 16 channels, then a
    downsampling block to 32 channels and one more block."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        down = nn.Sequential(nn.Conv2d(16, 32, 1, stride=2, bias=False), nn.BatchNorm2d(32))
        self.layer1 = nn.Sequential(BasicBlock(16, 16), BasicBlock(16, 16))
        self.layer2 = nn.Sequential(BasicBlock(16, 32, stride=2, downsample=down), BasicBlock(32, 32))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.layer2(self.layer1(self.relu(self.bn1(self.conv1(x)))))
        return self.fc(self.pool(x).flatten(1))


def block_colors(traced, colors_by_add):
    """Node name -> colour for every node inside a discovered block."""
    colors = {}
    for _, add, members, reason in discover_residual_regions(traced):
        color = colors_by_add.get(add.name, SKIPPED if reason else None)
        if color is None:
            continue
        for node in members:
            colors[node.name] = color
    return colors


def main():
    traced = TracedModel.create(SmallResNet(), (1, 3, 32, 32))
    found = discover_residual_regions(traced)
    valid = [(entry, add) for entry, add, _, reason in found if reason is None]
    colors_by_add = {add.name: PALETTE[i] for i, (_, add) in enumerate(valid)}
    draw_filtered_fx_graph(traced.gm, str(OUTPUT / '012-small-resnet-blocks-before'), fmt='png',
                           node_colors=block_colors(traced, colors_by_add), rankdir='TB')

    entry, add = valid[DUPLICATED_BLOCK]
    before_adds = {item[1].name for item in found}
    ResidualBlockAction(['duplicate', entry.name, add.name]).execute(traced)
    for _, new_add, _, _ in discover_residual_regions(traced):
        if new_add.name not in before_adds:  # the copy is the one new block: same colour as its source
            colors_by_add[new_add.name] = colors_by_add[add.name]
    draw_filtered_fx_graph(traced.gm, str(OUTPUT / '012-small-resnet-blocks-after'), fmt='png',
                           node_colors=block_colors(traced, colors_by_add), rankdir='TB')
    before = Image.open(OUTPUT / '012-small-resnet-blocks-before.png').convert('RGB')
    after = Image.open(OUTPUT / '012-small-resnet-blocks-after.png').convert('RGB')
    pair = Image.new('RGB', (before.width + after.width + 120, max(before.height, after.height)), 'white')
    pair.paste(before, (0, 0))
    pair.paste(after, (before.width + 120, 0))
    pair.save(OUTPUT / '012-small-resnet-blocks.png')
    print(f'{len(found)} blocks, {len(valid)} duplicable; duplicated the block ending at {add.name}')


if __name__ == '__main__':
    main()
