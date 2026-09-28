#!/usr/bin/env python3
"""Regenerate all report figures from released numerical source data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.figures import main
if __name__ == '__main__':
    main()
