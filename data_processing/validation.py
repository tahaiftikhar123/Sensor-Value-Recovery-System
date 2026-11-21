"""Identify sensor columns and create chronologically ordered numerical inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TIME_NAMES = {"timestamp", "datetime", "date_time", "date", "time", "recorded_at"}
METADATA_NAMES = {"machine_failure", "failure", "label", "target", "class", "status", "id"}
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "<na>"}


def normalized(name: str) -> str:
    return "_".join(name.strip().lower().split())


@dataclass
class ValidatedData:
    time_column: str
    timestamps: pd.Series
    hours: np.ndarray
    readings: pd.DataFrame
    warnings: list[str]


def validate_data(
    original: pd.DataFrame, time_column: str, excluded: tuple[str, ...] = (),
) -> ValidatedData:
    if time_column not in original:
        raise ValueError("Select a timestamp column to continue.")

    # UTC gives a common time basis for offsets. Naive timestamps use their recorded clock.
    timestamps = pd.to_datetime(
        original[time_column].str.strip(), format="mixed", utc=True, errors="coerce",
    )
    if timestamps.isna().any():
        first = timestamps.index[timestamps.isna()][0]
        raise ValueError(
            f"{time_column!r} contains an invalid or missing timestamp at data row {first + 1}."
        )

    duplicates = timestamps.duplicated(keep=False)
    if duplicates.any():
        example = original.loc[duplicates, time_column].iloc[0]
        raise ValueError(f"The time column contains duplicate timestamps, including {example!r}.")

    timestamps = timestamps.sort_values()
    hours = ((timestamps - timestamps.iloc[0]).dt.total_seconds() / 3600).to_numpy()
    if not np.isfinite(hours).all() or (np.diff(hours) <= 0).any():
        raise ValueError("The timestamps are too close or too far apart to represent as elapsed hours.")

    warnings = []
    if not timestamps.index.equals(original.index):
        warnings.append("Timestamps were sorted for interpolation. Downloads follow the original row order.")

    readings = pd.DataFrame(index=timestamps.index)
    for column in original:
        if column == time_column or column in excluded or normalized(column) in TIME_NAMES | METADATA_NAMES:
            continue
        cells = original[column].str.strip()
        missing = cells.str.casefold().isin(MISSING_TOKENS)
        numbers = pd.to_numeric(cells.mask(missing), errors="coerce").astype(float)
        finite = np.isfinite(numbers)
        invalid = ~missing & ~finite
        if not (finite.any() or missing.all() or np.isinf(numbers).any()):
            warnings.append(f"{column}: excluded because it contains no numerical readings.")
            continue
        readings[column] = numbers.where(finite).reindex(timestamps.index)
        if invalid.any():
            warnings.append(f"{column}: {invalid.sum()} invalid or non-finite readings counted as missing.")
        if not finite.any():
            warnings.append(f"{column}: no valid readings are available for interpolation.")
        elif finite.sum() == 1:
            warnings.append(f"{column}: at least two valid readings are needed for interpolation.")

    if readings.empty:
        raise ValueError("No numerical sensor columns were found. Check the selected time column and CSV headers.")

    return ValidatedData(time_column, timestamps, hours, readings, warnings)
