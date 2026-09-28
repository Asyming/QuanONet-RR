"""Physical coordinates for the released diffusion--reaction checkpoints."""
import numpy as np


def diffusion_model_coordinates(physical_coordinates):
    """Map (x, physical time) to (x, model time) for times in [0, 50/99]."""
    coordinates = np.asarray(physical_coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError('Coordinates must have shape (queries, 2).')
    if not np.isfinite(coordinates).all() or np.any(coordinates[:, 1] < 0) or np.any(coordinates[:, 1] > 50 / 99):
        raise ValueError('Physical times must lie in [0, 50/99].')
    model_coordinates = coordinates.copy()
    model_coordinates[:, 1] *= 99 / 50
    return model_coordinates.astype(np.float32)
