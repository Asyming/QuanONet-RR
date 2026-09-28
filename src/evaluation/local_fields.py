"""Predict dense extension solution fields and measure the manuscript local errors."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.evaluation.checkpoints import ROOT, load_model
from src.evaluation.local_metrics import evaluate
from src.training.run import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='solution_fields.npz produced by the extension generator')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--functions', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mq-workers', type=int, default=1)
    parser.add_argument('--eval-batch-size', type=int, default=100)
    args = parser.parse_args()
    config = json.loads((args.checkpoint / 'config.json').read_text())
    reference = args.reference or (ROOT / config['data_path']).parent / 'solution_fields.npz'
    with np.load(reference, allow_pickle=False) as archive:
        selected = np.argsort(archive['function_id'])[:args.functions]
        ids, branch, truth = [archive[key][selected] for key in ('function_id', 'u0', 'solution')]
        x, t = archive['x'], archive['t']
    branch, truth = branch.astype(np.float32), truth.astype(np.float32)
    coordinates = np.stack(np.meshgrid(x, t, indexing='ij'), axis=-1).reshape(-1, 2).astype(np.float32)
    shape_data = {}
    for split in ('train', 'test'):
        shape_data.update({split + '_branch_input': branch,
                           split + '_trunk_input': np.repeat(coordinates[:1], len(ids), axis=0),
                           split + '_output': truth[:, :1, :1].reshape(-1, 1)})
    model = load_model(args.checkpoint, config, shape_data, args.num_workers, args.mq_workers)
    quantum = config['simulator'] == 'mqvector'
    if quantum:
        import mindspore as ms
    else:
        import torch
    prediction = np.empty_like(truth)
    for function_index in range(len(ids)):
        values = prediction[function_index].reshape(-1, 1)
        for first in range(0, len(coordinates), args.eval_batch_size):
            last = min(first + args.eval_batch_size, len(coordinates))
            observed = np.repeat(branch[function_index:function_index + 1], last - first, axis=0)
            query = coordinates[first:last]
            if quantum:
                values[first:last] = model((ms.Tensor(observed), ms.Tensor(query))).asnumpy()
            else:
                with torch.no_grad():
                    values[first:last] = model((torch.as_tensor(observed), torch.as_tensor(query))).numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out / 'solution_predictions.npz'
    np.savez_compressed(prediction_path, truth=truth.reshape(-1, 1), prediction=prediction.reshape(-1, 1),
                        function_id=np.repeat(ids, len(coordinates)), unique_function_id=ids,
                        trunk_input=np.tile(coordinates, (len(ids), 1)))
    metrics = evaluate(reference, prediction_path, config['operator'])
    metrics.update(variant=config['variant'], seed=config['seed'], budget=config['budget'])
    atomic_json(args.out / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == '__main__':
    main()
