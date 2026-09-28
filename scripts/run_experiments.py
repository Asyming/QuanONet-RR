#!/usr/bin/env python3
"""Run selected entries from the frozen report experiment matrix."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', nargs='+', required=True)
    parser.add_argument('--operator', nargs='+')
    parser.add_argument('--variant', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--num_workers', type=int, default=1, help='CPU threads per model; selected runs execute sequentially')
    parser.add_argument('--mq-workers', type=int, default=1)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Classical models only; quantum models use CPU')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in (ROOT / 'configs/experiments.jsonl').read_text().splitlines()]
    known = {row['suite'] for row in rows}
    if set(args.suite) - known:
        parser.error('Available suites: ' + ', '.join(sorted(known)))
    selected = [row for row in rows if row['suite'] in args.suite
                and (not args.operator or row['operator'] in args.operator)
                and (not args.variant or row['variant'] in args.variant)
                and (args.seeds is None or row['seed'] in args.seeds)]
    if not selected:
        parser.error('No experiments match the filters.')
    for row in selected:
        flags = list(row['arguments'])
        for flag, value in [('--num_workers', args.num_workers), ('--mq-workers', args.mq_workers),
                            ('--device', 'cpu' if row['variant'].startswith(('TF', 'FF')) else args.device)]:
            if flag in flags:
                flags[flags.index(flag) + 1] = str(value)
            else:
                flags.extend([flag, str(value)])
        command = [sys.executable, '-u', str(ROOT / row['script']), *flags]
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)
    print(f'Selected experiments: {len(selected)}', flush=True)

if __name__ == '__main__':
    main()
