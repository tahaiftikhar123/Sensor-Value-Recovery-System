"""Evaluate both methods against the same held-out original readings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_processing.missing_values import METHODS, reconstruct


@dataclass
class Evaluation:
    held_out: int
    compared: int
    metrics: pd.DataFrame


def error_metrics(actual: np.ndarray, estimated: np.ndarray) -> tuple[float, float]:
    """Calculate MAE and RMSE for paired observations."""
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(actual - estimated)
        scale = errors.max()
        if scale == 0:
            return 0.0, 0.0
        if not np.isfinite(scale):
            # Scale before subtraction if the absolute error itself overflowed.
            scale = max(np.abs(actual).max(), np.abs(estimated).max())
            normalized = np.abs(actual / scale - estimated / scale)
        else:
            normalized = errors / scale
        return float(scale * normalized.mean()), float(scale * np.sqrt((normalized ** 2).mean()))


def evaluate(hours: np.ndarray, readings: np.ndarray, seed: int = 42) -> Evaluation:
    # Keep both endpoints visible so all held-out targets have bracketing readings.
    eligible = np.flatnonzero(np.isfinite(readings))[1:-1]
    count = max(1, round(0.1 * len(eligible))) if len(eligible) else 0
    targets = np.sort(np.random.default_rng(seed).choice(eligible, size=count, replace=False))
    visible = readings.copy()
    visible[targets] = np.nan
    actual = readings[targets]
    predictions = {method: reconstruct(hours, visible, method).values[targets] for method in METHODS}
    shared = np.isfinite(predictions["Linear"]) & np.isfinite(predictions["Lagrange"])
    compared = int(shared.sum())

    rows = []
    for method, estimates in predictions.items():
        mae, rmse = error_metrics(actual[shared], estimates[shared]) if compared else (np.nan, np.nan)
        rows.append({
            "Method": method,
            "Predictions": int(np.isfinite(estimates).sum()),
            "Evaluated": compared,
            "MAE": mae if np.isfinite(mae) else np.nan,
            "RMSE": rmse if np.isfinite(rmse) else np.nan,
        })
    return Evaluation(count, compared, pd.DataFrame(rows))
