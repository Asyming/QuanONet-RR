"""Rescore saved model predictions under the reference conventions used in Table 1 and Figure 3."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()
    if args.num_workers != 1:
        parser.error('This vectorized comparison uses one worker.')
    with np.load(args.reference) as reference, np.load(args.predictions) as fitted:
        for key in ('function_id', 'trunk_input'):
            if not np.array_equal(reference[key], fitted[key]):
                raise ValueError(f'Predictions and reference use different {key}.')
        if not np.array_equal(reference['original_labels'], fitted['truth']):
            raise ValueError('Predictions and reference use different original targets.')
        prediction = fitted['prediction'].astype(float)
        result = {}
        for key in ('original_labels', 'nominal_reference', 'scheme_aligned_reference'):
            truth = reference[key].astype(float)
            result[key] = float(np.linalg.norm(prediction - truth) / np.linalg.norm(truth))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
