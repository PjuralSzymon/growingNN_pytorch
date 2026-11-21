"""Estimate legal action-family counts by replaying recorded architecture edits.

Models are reinitialized, because the experiment does not save its trained
weights. Consequently this is a structural estimate, not the original pool.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.train_cifar10_exp010_mcts_meta import TinyCifar10Starter
from growingnn.actions.registry import generate_all_actions
from growingnn.core.config import RunningConfig
from growingnn.core.traced_model import TracedModel
from growingnn.actions.utils.layer_resize import resize_conv_output
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('board', type=Path)
    parser.add_argument('--diagnose-widening', action='store_true')
    args = parser.parse_args()
    traced = TracedModel.create(TinyCifar10Starter(), (1, 3, 32, 32))
    config = RunningConfig(generations=10, epochs=20, device='cpu')
    config.ACTIONS_ENABLE_WIDEN_CONV = True
    for path in sorted((args.board / 'simulations').glob('simulation_gen_*.json'),
                       key=lambda p: int(p.stem.split('_')[-1])):
        sim = json.loads(path.read_text(encoding='utf-8'))
        actions = generate_all_actions(traced, config)
        families = Counter(type(action).__name__ for action in actions)
        print(f"Generation {sim['generation']}: {len(actions)} available (replay estimate); "
              f"{sim['actionsAnalyzed']} evaluated; {dict(families)}")
        if args.diagnose_widening:
            for name, module in traced.gm.named_modules():
                if not isinstance(module, torch.nn.Conv2d) or module.groups != 1:
                    continue
                candidate = copy.deepcopy(traced)
                candidate.gm.eval()
                try:
                    with torch.no_grad():
                        probe = candidate.probe()
                        before = candidate.gm(probe)
                        resize_conv_output(candidate.gm, name, 2 * module.out_channels)
                        after = candidate.gm(probe)
                    print(f"  {name}: max output deviation {(before - after).abs().max().item():.6g}")
                except (RuntimeError, TypeError, ValueError) as exc:
                    print(f'  {name}: {type(exc).__name__}: {exc}')
        chosen = [action for action in actions if str(action) == sim.get('actionChosen')]
        if len(chosen) != 1:
            print('Replay stopped: chosen action could not be matched uniquely.')
            break
        chosen[0].execute(traced)


if __name__ == '__main__':
    main()
