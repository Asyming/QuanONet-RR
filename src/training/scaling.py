#!/usr/bin/env python3
"""Reproduce published latent-dimension scaling configurations with frozen selections."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "src/training/run.py"


def parameter_count(qubits, net_size):
    branch, branch_layers, trunk, trunk_layers = net_size
    return qubits * (3 * (branch * branch_layers + trunk * trunk_layers)
                     + 2 * (branch + trunk)) + 1


def classical_parameter_count(net_size, branch_inputs=100, trunk_inputs=1):
    branch_depth, branch_width, trunk_depth, trunk_width, latent = net_size
    count = 1
    for inputs, depth, width in ((branch_inputs, branch_depth, branch_width),
                                 (trunk_inputs, trunk_depth, trunk_width)):
        layers = [inputs] + [width] * (depth - 1) + [latent]
        count += sum((left + 1) * right for left, right in zip(layers, layers[1:]))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--num-qubits", type=int, required="--help" not in sys.argv)
    parser.add_argument("--net-size", nargs="+", type=int, required="--help" not in sys.argv)
    width, remaining = parser.parse_known_args()
    if "--help" in remaining:
        print("Scaling options: --num-qubits N --net-size N [N ...]")
    elif width.num_qubits < 2 or min(width.net_size) < 1:
        raise ValueError("Qubit count must be at least two and depths must be positive")
    spec = importlib.util.spec_from_file_location("rr_base_training", BASE)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_configuration = runner.configuration

    def configuration(args, data):
        if args.variant not in ("TFQuanONet", "DeepONet") or args.budget != 1 or args.operator != "Antideriv":
            raise ValueError("The scaling experiment uses TFQuanONet/DeepONet on Antiderivative")
        config = original_configuration(args, data)
        branch_inputs = data["train_branch_input"].shape[1]
        trunk_inputs = data["train_trunk_input"].shape[1]
        if args.variant == "TFQuanONet":
            if len(width.net_size) != 4:
                raise ValueError("TFQuanONet requires four circuit-depth entries")
            if width.num_qubits * width.net_size[0] < branch_inputs:
                raise ValueError("Branch encoding would truncate observed sensors")
            if width.num_qubits * width.net_size[2] < trunk_inputs:
                raise ValueError("Trunk encoding would truncate query coordinates")
            expected = parameter_count(width.num_qubits, width.net_size)
        else:
            if len(width.net_size) != 5 or width.net_size[4] != 2 ** width.num_qubits:
                raise ValueError("DeepONet requires an explicit output dimension equal to p = 2**n")
            expected = classical_parameter_count(width.net_size, branch_inputs, trunk_inputs)
        config.update(num_qubits=width.num_qubits, net_size=width.net_size,
                      latent_dimension=2 ** width.num_qubits,
                      expected_parameters=expected,
                      width_protocol="published_source_minimum_mean_test_mse_configurations_frozen_before_reruns")
        return config

    def pooled_training_evaluation(function):
        def train(args, config, data, deadline):
            model, state, completed = function(args, config, data, deadline)
            if completed and not args.train_only:
                metrics_path = args.out / "metrics.json"
                metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                if "train" not in metrics:
                    metrics["train"] = runner.evaluate(args, config, model, data, "train", 0.0)
                    runner.atomic_json(metrics_path, metrics)
            return model, state, completed
        return train

    runner.configuration = configuration
    runner.train_ms = pooled_training_evaluation(runner.train_ms)
    runner.train_classical = pooled_training_evaluation(runner.train_classical)
    sys.argv = [str(BASE), *remaining]
    runner.main()


if __name__ == "__main__":
    main()
