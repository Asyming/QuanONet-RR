#!/usr/bin/env python3
"""Aggregate full-test metrics across the five independent report repetitions."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]

def read_run(row):
    directory = ROOT / row['checkpoint']
    directory = directory.parent
    config = json.loads((directory / 'config.json').read_text())
    metric = json.loads((directory / 'metrics.json').read_text())['test']
    return row, config, metric

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/generated/test_summary.csv')
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / 'configs/experiments.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row['suite'] != 'training_curves']
    groups = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for row, config, metric in executor.map(read_run, rows):
            quantum = config['simulator'] == 'mqvector'
            latent = (2 ** config['num_qubits'] if config['model_type'] == 'QuanONet' else
                      (config['net_size'][-1] if len(config['net_size']) == 5 else min(config['net_size'][1], config['net_size'][3]))
                      if config['model_type'] == 'DeepONet' else None)
            key = (row['id'].split('/')[0], row['operator'], row['variant'], config['scale_coeff'],
                   config['num_train'], config['budget'], config['num_qubits'] if quantum else None,
                   latent, config.get('training_noise_fraction', 0.0))
            groups[key].append((row, config, metric))
    output = []
    fields = ('group', 'operator', 'variant', 'encoding_scale', 'training_functions', 'budget', 'num_qubits',
              'latent_dimension', 'training_noise_fraction')
    for key, values in sorted(groups.items()):
        if sorted(row['seed'] for row, _, _ in values) != list(range(5)):
            raise AssertionError(f'Expected all five repetitions: {key}')
        errors = [100 * metric['relative_l2'] for _, _, metric in values]
        parameters = {row['parameters'] for row, _, _ in values}
        if len(parameters) != 1:
            raise AssertionError(f'Inconsistent model sizes: {key}')
        output.append(dict(zip(fields, key), parameters=parameters.pop(),
                           mean_relative_l2_percent=statistics.mean(errors),
                           sample_sd_relative_l2_percent=statistics.stdev(errors), repetitions=5))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps(dict(runs=len(rows), configurations=len(output), output=str(args.out))))

if __name__ == '__main__':
    main()
