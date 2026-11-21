"""Validate manual readings and pass their time series to the shared recovery engine."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite

import numpy as np
import pandas as pd

from data_processing.missing_values import INVALID, RECORDED, RECOVERED, Recovery, reconstruct

PENDING = "Pending — Awaiting Additional Readings"
ESTIMATED = "Estimated"
UNABLE = "Unable to Estimate"
STATUS = {RECORDED: RECORDED, RECOVERED: ESTIMATED, INVALID: UNABLE}


@dataclass(frozen=True)
class Reading:
    time: float | datetime
    timestamp: str
    value: float | None


def elapsed_time(readings: tuple[Reading, ...]) -> np.ndarray:
    if not readings:
        return np.array([], dtype=float)

    first = readings[0].time
    if isinstance(first, datetime):
        hours = np.array([(entry.time - first).total_seconds() / 3600 for entry in readings])
    else:
        hours = np.array([entry.time - first for entry in readings])
    if not np.isfinite(hours).all() or (np.diff(hours) <= 0).any():
        raise ValueError("These times are too far apart or too close to resolve on one timeline.")
    return hours


def _finite_number(text: str, field: str) -> float:
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{field} must be a finite number.") from error
    if not isfinite(number):
        raise ValueError(f"{field} must be a finite number.")
    return number


def save_reading(
    readings: tuple[Reading, ...],
    timestamp: str,
    value: str | None,
    date_time: bool,
    *,
    replace: Reading | None = None,
) -> tuple[Reading, ...]:
    """Insert or replace an original reading after validating the full timeline."""
    timestamp = timestamp.strip()
    if date_time:
        try:
            parsed = datetime.fromisoformat(timestamp)
            time = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except (ValueError, OverflowError) as error:
            raise ValueError("Enter an ISO date-time such as 2026-11-21 14:30 or 2026-11-21T14:30+05:00.") from error
    else:
        time = _finite_number(timestamp, "Time")

    number = None if value is None else _finite_number(value, "Sensor reading")
    remaining = tuple(entry for entry in readings if entry != replace)
    if any(entry.time == time for entry in remaining):
        raise ValueError("This timestamp already has an entry. Use Edit readings to change it.")

    updated = tuple(sorted((*remaining, Reading(time, timestamp, number)), key=lambda entry: entry.time))
    elapsed_time(updated)
    return updated


def estimate_readings(
    readings: tuple[Reading, ...], method: str,
) -> tuple[np.ndarray, Recovery, pd.DataFrame]:
    original = np.array([entry.value for entry in readings], dtype=float)
    result = reconstruct(elapsed_time(readings), original, method)
    estimated = result.status == RECOVERED
    table = pd.DataFrame({
        "Timestamp": [entry.timestamp for entry in readings],
        "Original Reading": original,
        "Estimated Reading": np.where(estimated, result.values, np.nan),
        "Interpolation Method": [f"{method} Interpolation" if missing else "" for missing in np.isnan(original)],
        "Status": [STATUS.get(status, PENDING) for status in result.status],
    })
    return original, result, table
