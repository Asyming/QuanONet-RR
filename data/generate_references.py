"""Generate the two numerical-reference conventions used in Table 1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


def diffusion_reference(payload):
    forcing, times, output_x = payload
    points = forcing.shape[1]
    count = len(forcing)
    dx = 1 / (points - 1)
    scaled = np.linspace(0, 1, points) * (points - 1)
    left = np.minimum(np.floor(scaled).astype(int), points - 2)
    weight = scaled - left
    sampled_forcing = (forcing[:, left] * (1 - weight) + forcing[:, left + 1] * weight)[:, 1:-1]

    def rhs(t, values):
        interior = values.reshape(count, points - 2)
        padded = np.pad(interior, ((0, 0), (1, 1)))
        return (0.01 * (padded[:, 2:] - 2 * interior + padded[:, :-2]) / dx**2
                + 0.01 * interior**2 + sampled_forcing).ravel()

    result = solve_ivp(rhs, (0, times[-1]), np.zeros(count * (points - 2)),
                       t_eval=times, method='DOP853', rtol=1e-9, atol=1e-11)
    if not result.success:
        raise RuntimeError(result.message)
    values = np.pad(result.y.reshape(count, points - 2, -1), ((0, 0), (1, 1), (0, 0)))
    scaled = output_x * (points - 1)
    left = np.minimum(np.floor(scaled).astype(int), points - 2)
    weight = (scaled - left)[None, :, None]
    return values[:, left] * (1 - weight) + values[:, left + 1] * weight


def periodic_translation(initial, x, times, period):
    dx = 1 / (initial.shape[1] - 1)
    positions = np.remainder(x - times, period) / dx
    left = np.floor(positions).astype(int)
    fraction = positions - left
    right = (left + 1) % initial.shape[1]
    rows = np.arange(len(initial))[:, None]
    return initial[rows, left] * (1 - fraction) + initial[rows, right] * fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original dataset directory')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    operator = json.loads((args.source / 'manifest.json').read_text())['operator']
    if operator not in ('RDiffusion', 'Advection'):
        parser.error('References are defined for RDiffusion and Advection.')
    with np.load(args.source / 'dataset.npz') as data:
        ids, first = np.unique(data['test_function_id'], return_index=True)
        branch = data['test_branch_input'][first]
        trunk = data['test_trunk_input'].reshape(len(ids), -1, 2).astype(float)
        original = data['test_output']
        indices = data['test_query_index'].reshape(len(ids), -1)
    raw_path = args.source / 'released_cache/data' / f'{operator}_Operator_data' / f'{operator}_Operator_data_100_1.npz'
    with np.load(raw_path) as raw:
        all_initial = raw['u0_cals'].astype(float)
    # Recover the original float64 inputs before the dataset's float32 conversion.
    lookup = {row.astype(np.float32).tobytes(): i for i, row in enumerate(all_initial)}
    initial = all_initial[[lookup[row.tobytes()] for row in branch]]
    x_grid = np.linspace(0, 1, 100, dtype=np.float32).astype(float)
    spatial, temporal = indices // 100, indices % 100
    if operator == 'RDiffusion':
        dt = min((1 / 99)**2 / 0.02, 1 / 99)
        actual_times = np.arange(100) * max(1, int(1 // dt) // 100) * dt
        times = np.unique(np.concatenate((x_grid, actual_times)))
        blocks = np.array_split(initial, int(np.ceil(len(ids) / args.batch_size)))
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            fields = np.concatenate(list(pool.map(diffusion_reference, [(b, times, x_grid) for b in blocks])))
        rows = np.arange(len(ids))[:, None]
        nominal = fields[rows, spatial, np.searchsorted(times, x_grid)[temporal]]
        aligned = fields[rows, spatial, np.searchsorted(times, actual_times)[temporal]]
        conventions = {'nominal_reference': 'DOP853 on the 100-point grid at labelled times',
                       'scheme_aligned_reference': 'DOP853 on the same grid at actual frame times',
                       'rtol': 1e-9, 'atol': 1e-11}
    else:
        dt = 0.8 / 99
        actual_times = np.linspace(0, int(1 / dt) - 1, 100, dtype=int) * dt
        nominal = periodic_translation(initial, trunk[:, :, 0], trunk[:, :, 1], 1.0)
        aligned = periodic_translation(initial, trunk[:, :, 0], actual_times[temporal], 100 / 99)
        conventions = {'nominal_reference': 'Unit-speed periodic translation; period 1; labelled times',
                       'scheme_aligned_reference': 'Unit-speed periodic translation; period 100/99; actual frame times'}
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / 'reference_labels.npz', original_labels=original,
                        nominal_reference=nominal.reshape(-1, 1).astype(np.float32),
                        scheme_aligned_reference=aligned.reshape(-1, 1).astype(np.float32),
                        function_id=np.repeat(ids, trunk.shape[1]), unique_function_id=ids,
                        trunk_input=trunk.reshape(-1, 2).astype(np.float32))
    (args.output / 'settings.json').write_text(json.dumps(dict(operator=operator, **conventions), indent=2) + '\n')


if __name__ == '__main__':
    main()
