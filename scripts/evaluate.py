#!/usr/bin/env python3
"""Evaluate selected weights on full datasets, including sensor-noise experiments."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.checkpoints import load_model
from src.training import run as training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=['train', 'test', 'val'], default=['test'])
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mq-workers', type=int, default=1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--eval-batch-size', type=int, default=100)
    parser.add_argument('--noise-levels', type=float, nargs='+', default=[0.0])
    parser.add_argument('--noise-seed', type=int, default=20260923)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    args.variant = config['variant']
    args.train_functions = config['num_train']
    args.num_train, args.num_test, args.num_val = 1000, config['num_test'], 200
    data = training.load_arrays(args)
    args.out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, config, data, args.num_workers, args.mq_workers, args.device)
    metrics = {}
    for split in args.splits:
        for noise in (args.noise_levels if split == 'test' else [0.0]):
            tag = split if noise == 0 else f'{split}_noise{noise:g}'
            metrics[tag] = training.evaluate(args, config, model, data, split, noise)
    training.atomic_json(args.out / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
