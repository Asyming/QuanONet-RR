#!/usr/bin/env python3
"""Obtain the original QuanONet implementation used by the report."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd01934380a148c96526261ed99218b22c5097353'

def main():
    destination = ROOT / 'QuanONet'
    if destination.exists():
        print(f'Using existing checkout: {destination}. Report commit: {COMMIT}')
        return
    subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/Wang-Ruocheng/QuanONet.git', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', COMMIT], check=True)

if __name__ == '__main__':
    main()
