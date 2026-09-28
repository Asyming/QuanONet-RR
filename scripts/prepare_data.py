#!/usr/bin/env python3
"""Generate all report datasets and numerical references using the fixed seeds."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    'original': ['Antideriv', 'Homogeneous', 'Nonlinear', 'RDiffusion', 'Advection', 'Darcy'],
    'extension': ['Burgers', 'AllenCahn', 'Wave'],
    'references': ['RDiffusion', 'Advection'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=[*TASKS, 'all'], required=True)
    parser.add_argument('--operators', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(5)))
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--skip-existing', action='store_true', help='Skip outputs with all required data and completion files')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    suites = list(TASKS) if args.suite == 'all' else [args.suite]
    supported = {operator for suite in suites for operator in TASKS[suite]}
    if (args.operators and set(args.operators) - supported) or set(args.seeds) - set(range(5)):
        parser.error('Use supported operators and report repetition indices 0 to 4.')
    if args.num_workers < 1:
        parser.error('--num_workers must be positive.')
    output_root = args.output_root.resolve()
    for suite in suites:
        operators = [operator for operator in TASKS[suite] if not args.operators or operator in args.operators]
        for seed in args.seeds:
            for operator in operators:
                if suite == 'original':
                    script = ROOT / 'data/generate_original.py'
                    output = output_root / 'rr_original/main' / operator / f'seed{seed}'
                    flags = ['--operator', operator, '--seed', str(seed), '--output', str(output)]
                    required = [output / name for name in ('dataset.npz', 'dense.npz', 'manifest.json')]
                elif suite == 'extension':
                    script = ROOT / 'data/generate_extensions.py'
                    output = output_root / 'rr_extension/main_v1' / f'seed{seed}'
                    flags = ['--equation', operator, '--seed', str(202609230 + seed), '--output', str(output)]
                    required = [output / operator / name for name in ('dataset.npz', 'solution_fields.npz', 'manifest.json')]
                else:
                    script = ROOT / 'data/generate_references.py'
                    source = output_root / 'rr_original/main' / operator / f'seed{seed}'
                    output = output_root / 'references' / operator / f'seed{seed}'
                    flags = ['--source', str(source), '--output', str(output)]
                    required = [output / name for name in ('reference_labels.npz', 'settings.json')]
                command = [sys.executable, str(script), *flags, '--num_workers', str(args.num_workers)]
                if args.skip_existing and all(path.is_file() for path in required):
                    print(f'Skipping completed {suite}/{operator}/seed{seed}', flush=True)
                    continue
                if args.skip_existing and not args.dry_run and any(path.exists() for path in required):
                    raise FileExistsError(f'Incomplete dataset: {required[0].parent}. Use a fresh output directory.')
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
