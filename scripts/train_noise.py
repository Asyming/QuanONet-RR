#!/usr/bin/env python3
"""Run the noise training protocol."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.noise import main
if __name__ == "__main__":
    main()
