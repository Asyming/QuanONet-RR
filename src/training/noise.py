#!/usr/bin/env python3
"""Train the released operator models with online sensor-noise augmentation."""
import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/training/run.py"
spec = importlib.util.spec_from_file_location("rr_base_training", BASE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NoisyBranch:
    def __init__(self, clean, function_ids, std, seed):
        self.clean = clean
        self.function_ids = function_ids
        self.std = std
        self.generator = np.random.RandomState(seed)
        self.reads = 0
        self.shape = clean.shape

    def __getitem__(self, indices):
        values = self.clean[indices]
        if values.ndim != 2:
            raise ValueError("Augmentation requires a batch of sensor vectors")
        unique, inverse = np.unique(self.function_ids[indices], return_inverse=True)
        noise = self.generator.standard_normal((len(unique), self.shape[1])).astype(np.float32)
        self.reads += 1
        return values + np.float32(self.std) * noise[inverse]

    def state(self):
        return dict(rng=self.generator.get_state(), reads=self.reads)

    def restore(self, state):
        self.generator.set_state(state["rng"])
        self.reads = state["reads"]


def install(noise_args):
    original_configuration = runner.configuration
    original_ms = runner.train_ms
    original_classical = runner.train_classical
    original_atomic_pickle = runner.atomic_pickle

    def configuration(args, data):
        if args.variant not in ("TFQuanONet", "DeepONet"):
            raise ValueError("This experiment supports TFQuanONet and DeepONet only")
        config = original_configuration(args, data)
        first = np.unique(data["train_function_id"], return_index=True)[1]
        clean_std = float(np.std(data["train_branch_input"][first], dtype=np.float64))
        config.update(training_noise_fraction=noise_args.train_noise,
                      training_noise_std=noise_args.train_noise * clean_std,
                      training_noise_seed=noise_args.augmentation_seed + args.seed,
                      training_noise_protocol="Fresh independent Gaussian sensor noise per sampled function and minibatch; repeated function IDs share noise within a minibatch; clean outputs" if noise_args.train_noise else "Clean-input CPU control; no augmentation",
                      training_noise_reference="global_clean_training_branch_std")
        return config

    def train(args, config, data, deadline):
        clean = data["train_branch_input"]
        branch = NoisyBranch(clean, data["train_function_id"], config["training_noise_std"], config["training_noise_seed"])
        quantum = config["simulator"] == "mqvector"
        checkpoint = args.out / ("resume.pkl" if quantum else "resume.pt")
        if not quantum:
            import torch
            original_torch_save = torch.save
        if checkpoint.exists():
            if quantum:
                with checkpoint.open("rb") as handle:
                    saved = pickle.load(handle)
            else:
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            branch.restore(saved["augmentation"])

        def record(value):
            value = dict(value, augmentation=branch.state())
            return value

        def save_pickle(path, value):
            original_atomic_pickle(path, record(value))

        def save_torch(value, path, *positional, **keywords):
            if Path(path) == checkpoint.with_suffix(".pt.tmp"):
                value = record(value)
            original_torch_save(value, path, *positional, **keywords)

        # Save the independent augmentation stream atomically with the optimizer.
        if config["training_noise_fraction"]:
            data["train_branch_input"] = branch
        if quantum:
            runner.atomic_pickle = save_pickle
        else:
            torch.save = save_torch
        try:
            return (original_ms if quantum else original_classical)(args, config, data, deadline)
        finally:
            data["train_branch_input"] = clean
            if quantum:
                runner.atomic_pickle = original_atomic_pickle
            else:
                torch.save = original_torch_save

    runner.configuration = configuration
    runner.train_ms = train
    runner.train_classical = train


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--train-noise", type=float, default=0.05)
    parser.add_argument("--augmentation-seed", type=int, default=2026092400)
    noise_args, remaining = parser.parse_known_args()
    if noise_args.train_noise < 0:
        raise ValueError("Training noise must be nonnegative")
    sys.argv = [sys.argv[0], *remaining]
    install(noise_args)
    runner.main()


if __name__ == "__main__":
    main()
