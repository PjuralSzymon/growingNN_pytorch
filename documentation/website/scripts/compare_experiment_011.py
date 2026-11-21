"""Report completed Experiment 011 runs and matched-seed differences.

Pass a directory such as exp011_budgeted_mcts/v3/gen_10 and optionally a
baseline method. Incomplete runs have no run.json and are excluded.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', type=Path)
    parser.add_argument('--baseline', default='legacy')
    args = parser.parse_args()
    methods = defaultdict(dict)
    for path in args.runs.glob('*/cifar10/seed_*/run.json'):
        run = json.loads(path.read_text(encoding='utf-8'))
        methods[path.parents[2].name][int(run['seed'])] = run
    if not methods:
        print('No completed runs found.')
        return
    print('| Method | Seeds | Mean validation | Validation range | Mean test |')
    print('|---|---|---:|---:|---:|')
    for method, runs in sorted(methods.items()):
        val = [100 * r['final_validation_accuracy'] for r in runs.values()]
        test = [100 * r['final_test_accuracy'] for r in runs.values()]
        print(f"| {method} | {', '.join(str(s) for s in sorted(runs))} | "
              f"{statistics.mean(val):.2f}% | {min(val):.2f}–{max(val):.2f}% | "
              f"{statistics.mean(test):.2f}% |")
    baseline = methods.get(args.baseline, {})
    for method, runs in sorted(methods.items()):
        paired = sorted(runs.keys() & baseline.keys())
        if method == args.baseline or not paired:
            continue
        differences = [100 * (runs[s]['final_validation_accuracy']
                              - baseline[s]['final_validation_accuracy']) for s in paired]
        print(f"{method} vs {args.baseline}: mean paired validation difference "
              f"{statistics.mean(differences):+.2f} percentage points ({len(paired)} seeds).")


if __name__ == '__main__':
    main()
