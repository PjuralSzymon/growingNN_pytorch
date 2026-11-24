"""Experiment 014: Monte Carlo 2 growth on bigger datasets, CIFAR-100 and Tiny ImageNet.

Everything learned in Experiments 010-013 is kept: the BatchNorm + residual-block starter at 8 channels,
Monte Carlo 2 with growth actions only (widening, convolutions with BatchNorm, the add-BatchNorm action and
the residual-block actions), 2 measurements' worth of time per candidate (60 s, 5-25 min per search) and
the score as the mean of the last 2 scoring epochs. Only the dataset-dependent settings change:

* CIFAR-100: 50,000 training images of 32x32, 100 classes. Same image count and size as CIFAR-10, so the
  Experiment 013 Phase 18 recipe that learned CIFAR-10 well is kept: 10 generations of 30 epochs, 10
  scoring epochs on the CCS 10% simulation set.
* Tiny ImageNet: 100,000 training images of 64x64, 200 classes (the 10,000 official validation images are
  the test set). An epoch costs about 8x a CIFAR epoch at full resolution, so the starter's first
  convolution has stride 2 (64x64 -> 32x32 feature maps, about 2x a CIFAR epoch), generations are 15
  epochs, and scoring uses 5 epochs on a 5% stratified-random simulation set (5,000 images, the same
  size as the CIFAR sets). Stratified random avoids building the pixel-difficulty matrix of 90,000 x 12,288
  values that CCS would need.

Run (one dataset per process; results and the board are written live):
    python -u experiments/train_cifar10_exp014_bigger_datasets.py --dataset cifar100
    python -u experiments/train_cifar10_exp014_bigger_datasets.py --dataset tiny_imagenet
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import train_cifar10_exp013_action_replay as exp013  # noqa: E402
from growingnn.simulation.simulation_algorithms import montecarlo_alg_2  # noqa: E402

OUTPUT = ROOT / 'experiments/output/train_cifar10/runs/exp014_bigger_datasets/v1'

DATASETS = {
    'cifar100': {'classes': 100, 'stride': 1, 'generations': 10, 'epochs': 30, 'scoring_epochs': 10,
                 'simulation_set': 'ccs', 'simulation_percent': 10.0},
    'tiny_imagenet': {'classes': 200, 'stride': 2, 'generations': 10, 'epochs': 15, 'scoring_epochs': 5,
                      'simulation_set': 'stratified', 'simulation_percent': 5.0},
}


class BiggerDatasetStarter(nn.Module):
    """The Experiment 013 starter (conv -> BatchNorm -> ReLU, then relu(x + BN(conv(x))), pool, linear) for any
    number of classes; ``stride`` 2 halves the feature maps of 64x64 images."""

    def __init__(self, classes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(8)
        self.block_conv = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False)
        self.block_bn = nn.BatchNorm2d(8)
        self.linear2 = nn.Linear(8, classes, bias=False)

    def forward(self, x):
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        x = torch.relu(x + self.block_bn(self.block_conv(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        return self.linear2(x.flatten(1))


def spec(dataset: str) -> dict:
    """The Experiment 013 Phase 18 Monte Carlo 2 spec, adapted to ``dataset``."""
    values = DATASETS[dataset]
    base = exp013.comparison_spec(montecarlo_alg_2)
    note = (f"Experiment 014: Monte Carlo 2 growth on {dataset}: {values['generations']} generations of "
            f"{values['epochs']} epochs, {values['scoring_epochs']} scoring epochs on a "
            f"{values['simulation_percent']:g}% {values['simulation_set']} simulation set, starter stride "
            f"{values['stride']}.")
    return {**base, 'dataset': dataset, 'folder': dataset, 'phase_name': f'exp014_{dataset}',
            'label': f'014_bigger_{dataset}', 'name': f'Experiment 014 ({dataset})',
            'id_prefix': f'exp014_{dataset}', 'note': note, 'generations': values['generations'],
            'epochs': values['epochs'], 'scoring_epochs': values['scoring_epochs'],
            'simulation_set': values['simulation_set'],
            'overrides': {**base.get('overrides', {}), 'simulation_set_percent': values['simulation_percent']},
            'starter': lambda: BiggerDatasetStarter(values['classes'], values['stride']),
            'starter_name': f"BatchNorm + residual-block starter, 8 channels, {values['classes']} classes, "
                            f"first-convolution stride {values['stride']}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--dataset', choices=sorted(DATASETS), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1])
    args = parser.parse_args()
    exp013.run_full_growth(OUTPUT, args.seeds, spec=spec(args.dataset))


if __name__ == '__main__':
    main()
