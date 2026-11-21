"""Compare completed Experiment 011 and 012 runs on matching seeds.

Inputs are method directories containing cifar10/seed_*/run.json. The script
does not imply that runs on different source versions or devices are matched.
"""
import argparse
import json
from pathlib import Path
import statistics


def load_runs(method_dir):
    runs = {}
    for path in method_dir.glob('cifar10/seed_*/run.json'):
        row = json.loads(path.read_text(encoding='utf-8'))
        seed = int(row['seed'])
        if seed in runs:
            raise ValueError(f'duplicate seed {seed} in {method_dir}')
        runs[seed] = row
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    baseline, candidate = load_runs(args.baseline), load_runs(args.candidate)
    seeds = sorted(baseline.keys() & candidate.keys())
    if not seeds:
        print('No matched completed seeds.')
        return
    print('| Seed | Baseline validation | Candidate validation | Difference (pp) |')
    print('|---:|---:|---:|---:|')
    differences = []
    for seed in seeds:
        a = 100 * baseline[seed]['final_validation_accuracy']
        b = 100 * candidate[seed]['final_validation_accuracy']
        differences.append(b - a)
        print(f'| {seed} | {a:.2f}% | {b:.2f}% | {b - a:+.2f} |')
    print(f'\nMean paired validation difference: {statistics.mean(differences):+.2f} pp '
          f'over {len(seeds)} seed(s).')
    print('Verify source versions, device, and training recipe before interpreting this comparison.')


if __name__ == '__main__':
    main()
