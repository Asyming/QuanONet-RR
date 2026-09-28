#!/usr/bin/env python3
"""Compute local solution errors on dense PDE grids."""

import numpy as np


def masked_rmse(error, mask):
    counts = mask.sum(axis=(1, 2))
    values = np.full(len(error), np.nan)
    np.divide(np.sum(error**2 * mask, axis=(1, 2)), counts, out=values, where=counts > 0)
    return np.sqrt(values), counts


def evaluate(reference_path, prediction_path, equation):
    with np.load(reference_path) as reference, np.load(prediction_path) as fitted:
        source_ids = reference["function_id"]
        ids = fitted["unique_function_id"]
        lookup = {int(function_id): index for index, function_id in enumerate(source_ids)}
        indices = [lookup[int(function_id)] for function_id in ids]
        truth = reference["solution"][indices].astype(float)
        prediction = fitted["prediction"].reshape(truth.shape).astype(float)
        x, t = reference["x"], reference["t"]
    error = prediction - truth
    full_rmse = np.sqrt(np.mean(error**2, axis=(1, 2)))
    if equation == "Burgers":
        gradient = np.abs((np.roll(truth, -1, axis=1) - np.roll(truth, 1, axis=1)) / (2 * (x[1] - x[0])))
        active = np.broadcast_to(t[None, None, :] > 0, truth.shape)
        threshold = np.quantile(gradient[:, :, t > 0].reshape(len(truth), -1), 0.9, axis=1)
        mask = active & (gradient >= threshold[:, None, None])
        region = "Largest 10% reference spatial gradients over t>0, separately per function"
    elif equation == "AllenCahn":
        mask = (np.abs(truth) < 0.9) & (t[None, None, :] >= 0.5)
        region = "Reference |u|<0.9 at t>=0.5"
    else:
        spatial_index = int(np.argmin(np.abs(x - 0.2)))
        mask = np.zeros(truth.shape, dtype=bool)
        mask[:, spatial_index, :] = True
        region = f"Fixed spatial probe x={x[spatial_index]:.8g}"
    local_rmse, counts = masked_rmse(error, mask)
    per_function = [{"function_id": int(function_id), "global_rmse": float(global_value),
                     "local_rmse": float(local_value) if count else None,
                     "local_points": int(count)}
                    for function_id, global_value, local_value, count in zip(ids, full_rmse, local_rmse, counts)]
    return {"equation": equation, "reference": str(reference_path), "prediction": str(prediction_path),
            "region": region, "metric": "absolute RMSE",
            "functions": len(truth), "reference_functions": len(source_ids), "global_rmse_mean": float(full_rmse.mean()),
            "local_rmse_mean": float(np.mean(local_rmse[counts > 0])) if np.any(counts > 0) else None,
            "excluded_empty_regions": int(np.sum(counts == 0)), "per_function": per_function}
