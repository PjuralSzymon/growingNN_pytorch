"""Summarize completed generation/search records from Experiment 011.

Usage: python summarize_experiment_011.py path/to/seed_1/board
Partial runs are supported; each line reports only an existing generation.
"""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('board', type=Path)
    parser.add_argument('--candidates', action='store_true',
                        help='Show per-candidate scores and visit counts for each search')
    args = parser.parse_args()
    print('| Gen | Validation | Parameters | Search sec | Rollouts | Candidates | Chosen |')
    print('|---:|---:|---:|---:|---:|---:|---|')
    for path in sorted((args.board / 'generations').glob('generation_*.json'),
                       key=lambda p: int(p.stem.split('_')[-1])):
        generation = read(path)
        number = generation['generation']
        sim_path = args.board / 'simulations' / f'simulation_gen_{number}.json'
        simulation = read(sim_path) if sim_path.exists() else {}
        chosen = next((c.get('name') for c in simulation.get('candidates', [])
                       if c.get('chosen')), '—')
        val = generation.get('finalValAcc')
        val = f'{100 * val:.2f}%' if val is not None else '—'
        print(f"| {number} | {val} | {generation.get('paramCount', '—')} | "
              f"{simulation.get('durationSec', '—')} | {simulation.get('rollouts', '—')} | "
              f"{simulation.get('actionsAnalyzed', '—')} | {chosen} |")
    if args.candidates:
        for path in sorted((args.board / 'simulations').glob('simulation_gen_*.json'),
                           key=lambda p: int(p.stem.split('_')[-1])):
            simulation = read(path)
            print(f"\nGeneration {simulation['generation']} candidates:")
            for candidate in simulation.get('candidates', []):
                score = candidate.get('score')
                score = f'{100 * score:.2f}%' if isinstance(score, (float, int)) else '—'
                mark = '*' if candidate.get('chosen') else ' '
                print(f"{mark} {score:>8}  visits={candidate.get('visits', 0):<3} "
                      f"{candidate.get('action', candidate.get('name', 'unknown'))}")


if __name__ == '__main__':
    main()
