"""Detect gaps, reconstruct readings, and assemble export tables."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_processing.validation import ValidatedData
from numerical_methods.lagrange_interpolation import lagrange_interpolate, local_indices
from numerical_methods.linear_interpolation import linear_interpolate

METHODS = ("Linear", "Lagrange")
RECORDED = "Recorded"
RECOVERED = "Recovered"
BOUNDARY = "Outside known time range"
INSUFFICIENT = "Insufficient known readings"
INVALID = "Numerically invalid estimate"


@dataclass
class Recovery:
    values: np.ndarray
    status: np.ndarray


def sensor_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def missing_summary(readings: pd.DataFrame) -> pd.DataFrame:
    missing = readings.isna().sum()
    return pd.DataFrame({
        "Sensor": [sensor_label(name) for name in readings],
        "Total": len(readings),
        "Available": (len(readings) - missing).to_numpy(),
        "Missing": missing.to_numpy(),
        "Missing %": (100 * missing / len(readings)).to_numpy(),
    })


def reconstruct(hours: np.ndarray, readings: np.ndarray, method: str) -> Recovery:
    if method not in METHODS:
        raise ValueError(f"Unknown interpolation method: {method}")
    if len(hours) != len(readings) or not np.isfinite(hours).all() or (np.diff(hours) <= 0).any():
        raise ValueError("Interpolation requires distinct, sorted elapsed hours for all readings.")

    values = np.asarray(readings, dtype=float).copy()
    known = np.isfinite(values)
    values[~known] = np.nan
    status = np.full(len(values), RECORDED, dtype=object)
    known_hours, known_values = hours[known], values[known]
    if len(known_hours) < 2:
        status[~known] = INSUFFICIENT
        return Recovery(values, status)

    gaps = np.flatnonzero(~known)
    positions = np.searchsorted(known_hours, hours[gaps])
    for index, right in zip(gaps, positions):
        if right == 0 or right == len(known_hours):
            status[index] = BOUNDARY
            continue

        x = float(hours[index])
        try:
            if method == "Linear":
                x0, x1 = map(float, known_hours[right - 1:right + 1])
                y0, y1 = map(float, known_values[right - 1:right + 1])
                estimate = linear_interpolate(x, x0, y0, x1, y1)
            else:
                anchors = local_indices(known_hours, x, right)
                estimate = lagrange_interpolate(x, known_hours[anchors], known_values[anchors])
        except (ArithmeticError, ValueError):
            estimate = np.nan

        if np.isfinite(estimate):
            values[index] = estimate
            status[index] = RECOVERED
        else:
            status[index] = INVALID
    return Recovery(values, status)


def sensor_results(
    original: pd.DataFrame, data: ValidatedData, sensor: str, results: dict[str, Recovery]
) -> pd.DataFrame:
    indices = data.readings.index
    readings = data.readings[sensor]
    recorded = original.loc[indices, sensor].where(readings.notna(), "")
    table = pd.DataFrame({
        "Timestamp": original.loc[indices, data.time_column],
        "Original": recorded,
    })
    for method, result in results.items():
        completed = recorded.copy()
        recovered = result.status == RECOVERED
        completed.iloc[recovered] = result.values[recovered]
        table[method] = completed
        table[f"{method} status"] = result.status

    if len(results) == 2:
        linear, lagrange = results["Linear"], results["Lagrange"]
        with np.errstate(over="ignore", invalid="ignore"):
            difference = lagrange.values - linear.values
        table["Lagrange − Linear"] = np.where(
            (linear.status == RECOVERED) & (lagrange.status == RECOVERED) & np.isfinite(difference),
            difference,
            np.nan,
        )
    return table.reindex(original.index)


def recover_all(
    original: pd.DataFrame, data: ValidatedData, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = original.copy()
    summary = []
    for sensor in data.readings:
        readings = data.readings[sensor]
        result = reconstruct(data.hours, readings.to_numpy(), method)
        missing = readings.isna().to_numpy()
        recovered = result.status == RECOVERED
        completed.loc[readings.index[missing], sensor] = ""
        completed.loc[readings.index[recovered], sensor] = result.values[recovered]
        summary.append({
            "Sensor": sensor_label(sensor),
            "Missing Before": int(missing.sum()),
            "Recovered": int(recovered.sum()),
            "Remaining Missing": int((missing & ~recovered).sum()),
            "Numerically Invalid": int((result.status == INVALID).sum()),
        })
    return completed, pd.DataFrame(summary)
