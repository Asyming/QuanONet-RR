#!/usr/bin/env python3
"""Generate periodic multi-initial-condition PDE datasets for QuanONet."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy


@dataclass(frozen=True)
class Equation:
    name: str
    length: float = 1.0
    final_time: float = 1.0
    diffusivity: float = 0.0
    reaction_rate: float = 0.0
    wave_speed: float = 0.0


EQUATIONS = {
    "Burgers": Equation("Burgers", diffusivity=0.01),
    "AllenCahn": Equation("AllenCahn", diffusivity=0.001, reaction_rate=5.0),
    "Wave": Equation("Wave", wave_speed=1.0),
}
SPLITS = {"train": 0, "val": 1, "test": 2}


def initial_coefficients(seed, split, function_ids, modes, length_scale, length):
    frequencies = 2 * np.pi * np.arange(1, modes + 1) / length
    weights = np.exp(-0.5 * (frequencies * length_scale) ** 2)
    weights /= weights.sum()
    coefficients = np.stack([
        np.random.default_rng(np.random.SeedSequence([seed, SPLITS[split], int(i)]))
        .normal(size=(2, modes))
        for i in function_ids
    ])
    return coefficients * np.sqrt(weights)[None, None, :]


def evaluate_initial(coefficients, x, length):
    phase = 2 * np.pi * np.outer(np.arange(1, coefficients.shape[-1] + 1), x / length)
    return coefficients[:, 0] @ np.cos(phase) + coefficients[:, 1] @ np.sin(phase)


def etdrk4_coefficients(linear, dt):
    # Contour means avoid cancellation near zero eigenvalues.
    roots = np.exp(1j * np.pi * (np.arange(1, 33) - 0.5) / 32)
    z = dt * linear[:, None] + roots[None, :]
    q = dt * np.real(np.mean((np.exp(z / 2) - 1) / z, axis=1))
    f1 = dt * np.real(np.mean((-4 - z + np.exp(z) * (4 - 3 * z + z**2)) / z**3, axis=1))
    f2 = dt * np.real(np.mean((2 + z + np.exp(z) * (-2 + z)) / z**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * z - z**2 + np.exp(z) * (4 - z)) / z**3, axis=1))
    return np.exp(dt * linear), np.exp(dt * linear / 2), q, f1, f2, f3


def solve(coefficients, equation, solver_n, output_n, times, max_dt):
    if solver_n % output_n:
        raise ValueError("The output grid must be an exact subset of the solver grid")
    x = np.arange(solver_n) * equation.length / solver_n
    initial = evaluate_initial(coefficients, x, equation.length)
    state = np.fft.rfft(initial, axis=-1)
    wavenumber = 2 * np.pi * np.fft.rfftfreq(solver_n, equation.length / solver_n)
    output = np.empty((len(coefficients), output_n, len(times)), dtype=np.float64)
    stride = solver_n // output_n
    output[:, :, 0] = initial[:, ::stride]
    if equation.name == "Wave":
        for index, t in enumerate(times[1:], 1):
            spectral = state * np.cos(equation.wave_speed * wavenumber * t)
            output[:, :, index] = np.fft.irfft(spectral, n=solver_n)[:, ::stride]
        return output

    interval = times[1] - times[0]
    steps = int(np.ceil(interval / max_dt))
    dt = interval / steps
    linear = -equation.diffusivity * wavenumber**2 + equation.reaction_rate
    e, e2, q, f1, f2, f3 = etdrk4_coefficients(linear, dt)
    cutoff = solver_n // (3 if equation.name == "Burgers" else 4)
    mask = np.arange(len(wavenumber)) < cutoff
    state *= mask

    def nonlinear(value):
        physical = np.fft.irfft(value * mask, n=solver_n)
        if equation.name == "Burgers":
            transformed = -0.5j * wavenumber * np.fft.rfft(physical**2, axis=-1)
        else:
            transformed = -equation.reaction_rate * np.fft.rfft(physical**3, axis=-1)
        return transformed * mask

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for index in range(1, len(times)):
            for _ in range(steps):
                nv = nonlinear(state)
                a = e2 * state + q * nv
                na = nonlinear(a)
                b = e2 * state + q * na
                nb = nonlinear(b)
                c = e2 * a + q * (2 * nb - nv)
                state = e * state + f1 * nv + 2 * f2 * (na + nb) + f3 * nonlinear(c)
            output[:, :, index] = np.fft.irfft(state, n=solver_n)[:, ::stride]
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite reference solution")
    return output


def solve_chunk(payload):
    coefficients, equation, solver_n, output_n, times, max_dt = payload
    return solve(coefficients, equation, solver_n, output_n, times, max_dt)








def encode_split(u0, solution, coords, ids, query_count, seed, split):
    count = len(u0)
    query_indices = np.stack([
        np.random.default_rng(np.random.SeedSequence([seed, 17, SPLITS[split], int(i)]))
        .choice(len(coords), query_count, replace=False)
        for i in ids
    ])
    return {
        f"{split}_branch_input": np.repeat(u0, query_count, axis=0).astype(np.float32),
        f"{split}_trunk_input": coords[query_indices].reshape(-1, 2).astype(np.float32),
        f"{split}_output": solution.reshape(count, -1)[np.arange(count)[:, None], query_indices].reshape(-1, 1).astype(np.float32),
        f"{split}_function_id": np.repeat(ids, query_count),
        f"{split}_query_index": query_indices,
    }


def generate(equation, args):
    start = time.perf_counter()
    destination = args.output / equation.name
    destination.mkdir(parents=True, exist_ok=False)
    x = np.arange(args.output_nx) * equation.length / args.output_nx
    sensors = np.arange(args.sensors) * equation.length / args.sensors
    times = np.linspace(0, equation.final_time, args.output_nt)
    coords = np.stack(np.meshgrid(x, times, indexing="ij"), axis=-1).reshape(-1, 2)
    all_encoded = {}
    files = {}
    split_counts = {"train": args.num_train, "val": args.num_val, "test": args.num_test}
    for split, count in split_counts.items():
        if count == 0:
            continue
        ids = np.arange(count, dtype=np.int64) + SPLITS[split] * 10000000
        coefficients = initial_coefficients(args.seed, split, ids, args.modes, args.length_scale, equation.length)
        u0 = evaluate_initial(coefficients, sensors, equation.length)
        payloads = [(part, equation, args.solver_n, args.output_nx, times, args.max_dt)
                    for part in np.array_split(coefficients, int(np.ceil(count / args.batch_size)))]
        if args.num_workers == 1:
            solutions = [solve_chunk(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                solutions = list(executor.map(solve_chunk, payloads))
        solution = np.concatenate(solutions, axis=0)
        path = destination / f"{split}_functions.npz"
        np.savez_compressed(path, u0=u0.astype(np.float32), solution=solution.astype(np.float32),
                            coefficients=coefficients, function_id=ids, x=x, t=times, sensors=sensors)
        files[path.name] = {"bytes": path.stat().st_size}
        queries = args.train_queries if split == "train" else args.test_queries
        all_encoded.update(encode_split(u0, solution, coords, ids, queries, args.seed, split))
        if split == "test":
            field_count = min(count, args.field_functions)
            field_solution = solve(coefficients[:field_count], equation, args.solver_n,
                               args.field_nx, times, args.max_dt)
            field_path = destination / "solution_fields.npz"
            np.savez_compressed(field_path, u0=u0[:field_count].astype(np.float32),
                                solution=field_solution.astype(np.float32),
                                coefficients=coefficients[:field_count], function_id=ids[:field_count],
                                x=np.arange(args.field_nx) * equation.length / args.field_nx,
                                t=times, sensors=sensors)
            files[field_path.name] = {"bytes": field_path.stat().st_size}
        print(json.dumps({"equation": equation.name, "split": split, "functions": count,
                          "solution_min": float(solution.min()), "solution_max": float(solution.max()),
                          "elapsed_seconds": time.perf_counter() - start}), flush=True)
    cache = destination / "dataset.npz"
    np.savez_compressed(cache, **all_encoded)
    files[cache.name] = {"bytes": cache.stat().st_size}
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "equation": asdict(equation), "boundary": "periodic",
        "wave_initial_velocity": "zero" if equation.name == "Wave" else None,
        "initial_condition": {"distribution": "zero-spatial-mean periodic Gaussian random field",
                              "length_scale": args.length_scale, "pointwise_variance": 1.0,
                              "fourier_modes": args.modes,
                              "coefficient_variance": "exp(-0.5*(2*pi*k*ell/L)^2) / sum_k exp(-0.5*(2*pi*k*ell/L)^2)",
                              "seed": args.seed, "generator": "NumPy PCG64 with independent split/function SeedSequence"},
        "solver": {"method": "exact Fourier propagation" if equation.name == "Wave" else "Fourier pseudospectral ETDRK4",
                   "dtype": "float64/complex128", "grid_points": args.solver_n, "max_dt": args.max_dt,
                   "nonlinear_cutoff_fraction_of_nyquist": None if equation.name == "Wave" else (2/3 if equation.name == "Burgers" else 1/2)},
        "observations": {"sensors": args.sensors, "output_nx": args.output_nx, "output_nt": args.output_nt,
                         "x_endpoint_included": False, "t_endpoints_included": True,
                         "spatial_resampling": "exact subset of the solver grid", "flattening": "x-major, time-minor",
                         "train_queries_per_function": args.train_queries, "val_test_queries_per_function": args.test_queries},
        "solution_fields": {"first_test_functions": min(args.num_test, args.field_functions),
                        "space_points": args.field_nx, "time_points": args.output_nt,
                        "included_in_primary_test_metric": False},
        "split_functions": split_counts, "files": files,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "wall_seconds": time.perf_counter() - start,
        "resources": {"workers": args.num_workers, "batch_size": args.batch_size},
        "citations": ["https://doi.org/10.1038/s42256-024-00886-8", "https://doi.org/10.1038/s42256-026-01265-1",
                      "https://github.com/Ceyron/exponax", "https://doi.org/10.1137/S1064827502410633"],
        "scope_note": "New periodic initial-condition-to-solution applications; these configurations do not reproduce the cited NMI datasets.",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"equation": equation.name, "complete": True, "wall_seconds": manifest["wall_seconds"]}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--equation", choices=["all", *EQUATIONS], default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_train", type=int, default=1000)
    parser.add_argument("--num_val", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--sensors", type=int, default=100)
    parser.add_argument("--output_nx", type=int, default=100)
    parser.add_argument("--output_nt", type=int, default=100)
    parser.add_argument("--solver_n", type=int, default=800)
    parser.add_argument("--field_nx", type=int, default=400)
    parser.add_argument("--field_functions", type=int, default=16)
    parser.add_argument("--max_dt", type=float, default=0.0005)
    parser.add_argument("--length_scale", type=float, default=0.1)
    parser.add_argument("--modes", type=int, default=32)
    parser.add_argument("--train_queries", type=int, default=100)
    parser.add_argument("--test_queries", type=int, default=1000)
    args = parser.parse_args()
    if args.output_nt < 2 or args.modes >= min(args.sensors, args.output_nx) // 2:
        parser.error("At least two times and fewer than Nyquist initial modes are required")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    for name in EQUATIONS if parsed.equation == "all" else [parsed.equation]:
        generate(EQUATIONS[name], parsed)
