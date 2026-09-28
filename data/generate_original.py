import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'QuanONet'))
from data_utils import data_generation as dg
from data_utils.data_manager import GENERATOR_MAP




class CachedGRF:
    def __init__(self):
        self.factors = {}

    def __call__(self, m, length_scale=0.2):
        if length_scale not in self.factors:
            x = np.linspace(0, 1, 1024)
            covariance = dg.rbf(x[:, None], x[:, None], (1.0, length_scale))
            self.factors[length_scale] = (x, np.linalg.cholesky(covariance + 1e-10 * np.eye(1024)))
        x, factor = self.factors[length_scale]
        values = factor @ np.random.randn(1024)
        return lambda query: np.interp(query, x, values), np.interp(np.linspace(0, 1, m), x, values)


def encode(branch, solution, coordinates, queries, rng, id_offset):
    n = len(branch)
    indices = np.stack([rng.choice(len(coordinates), queries, replace=False) for _ in range(n)])
    flat = solution.reshape(n, -1)
    return {
        'branch_input': np.repeat(branch, queries, axis=0).astype(np.float32),
        'trunk_input': coordinates[indices.reshape(-1)].astype(np.float32),
        'output': flat[np.arange(n)[:, None], indices].reshape(-1, 1).astype(np.float32),
        'function_id': np.repeat(np.arange(n, dtype=np.int32) + id_offset, queries),
        'query_index': indices.reshape(-1).astype(np.int32),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--operator', choices=['Antideriv', 'Homogeneous', 'Nonlinear', 'RDiffusion', 'Advection', 'Darcy'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=1000)
    parser.add_argument('--num_val', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'dataset.npz'
    manifest_path = output / 'manifest.json'
    if target.exists():
        raise FileExistsError(f'Dataset already exists: {target}')
    cached = CachedGRF()
    dg.generate_random_gaussian_field = cached
    dg.cpu_count = lambda: args.num_workers
    is_ode = args.operator in dg.ODE_SYSTEMS
    points = 25 if args.operator == 'Darcy' else 100
    num_cal = 1000 if is_ode else 100
    train_queries = 10 if is_ode else 100
    test_queries = 100 if is_ode else (500 if args.operator == 'Darcy' else 1000)
    if args.num_train < 100:
        test_queries = min(test_queries, 100)
    work = output / 'released_cache'
    work.mkdir(exist_ok=True)
    os.chdir(work)
    data_seed = 202609230 + args.seed
    np.random.seed(data_seed)
    generated = GENERATOR_MAP[args.operator](args.num_train, args.num_test, points, 100, num_cal=num_cal)
    train_branch, train_solution, test_branch, test_solution = generated[:4]
    if len(train_branch) != args.num_train or len(test_branch) != args.num_test:
        raise AssertionError('Released generator returned an incomplete function cohort')
    axes = generated[4:]
    coordinates = axes[0][:, None] if is_ode else np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 2)
    # Keep the released RNG stream and its query-sampling order for the main split.
    train = encode(train_branch, train_solution, coordinates, train_queries, np.random, 0)
    test = encode(test_branch, test_solution, coordinates, test_queries, np.random, args.num_train)
    arrays = {f'{split}_{key}': value for split, block in [('train', train), ('test', test)] for key, value in block.items()}
    if args.num_val:
        val_work = output / 'validation_cache'
        val_work.mkdir(exist_ok=True)
        os.chdir(val_work)
        np.random.seed(data_seed + 100000)
        count_a = args.num_val // 2
        val_result = GENERATOR_MAP[args.operator](count_a, args.num_val-count_a, points, 100, num_cal=num_cal)
        val_branch = np.concatenate([val_result[0], val_result[2]])
        val_solution = np.concatenate([val_result[1], val_result[3]])
        val = encode(val_branch, val_solution, coordinates, min(100, len(coordinates)), np.random, args.num_train + args.num_test)
        arrays.update({f'val_{key}': value for key, value in val.items()})
    for key, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f'Nonfinite values in {key}')
    partial = output / 'dataset.partial.npz'
    np.savez_compressed(partial, **arrays)
    os.replace(partial, target)
    dense = output / 'dense.npz'
    np.savez_compressed(dense, branch=test_branch[:16], solution=test_solution[:16], coordinates=coordinates,
                        function_id=np.arange(min(16, args.num_test)) + args.num_train,
                        grid_shape=np.asarray(test_solution.shape[1:]),
                        axes_names=np.asarray(['x'] if is_ode else (['x', 'y'] if args.operator == 'Darcy' else ['x', 't'])))
    manifest = {'operator': args.operator, 'protocol': 'as_released', 'data_seed': data_seed,
                'model_seed_pair': args.seed, 'num_train': args.num_train, 'num_test': args.num_test,
                'num_val': args.num_val, 'train_queries_per_function': train_queries,
                'test_queries_per_function': test_queries, 'num_points': points, 'num_points_0': 100,
                'num_cal': num_cal, 'length_scale': 0.2, 'validation_seed': data_seed + 100000,
                'num_workers': args.num_workers, 'host': platform.node(),
                'seconds': time.perf_counter() - started, 'schema': {key: list(value.shape) for key, value in arrays.items()},
                'notes': ['Cached only the deterministic GRF Cholesky factor; original RNG draws and solvers retained.',
                          'Five data seeds are explicit independent repeats, paired across models.',
                          'Validation curves do not select the reported checkpoint.',
                          'As-released PDE time labels and numerical discretizations are retained and disclosed.']}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'complete', 'output': str(output), 'seconds': manifest['seconds']}), flush=True)


if __name__ == '__main__':
    main()
