"""Read CSV records while retaining the exact cell text for display and export."""

import csv
from io import StringIO
from pathlib import Path

import pandas as pd

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "datasets" / "iiot_all_sensors_missing_values.csv"


def load_csv(data: bytes) -> pd.DataFrame:
    try:
        reader = csv.reader(StringIO(data.decode("utf-8-sig"), newline=""), strict=True)
        columns = next(reader, [])
        names = [name.strip().casefold() for name in columns]
        if not names or not all(names):
            raise ValueError("The CSV needs a header with a name for every column.")
        if len(names) != len(set(names)):
            raise ValueError("The CSV contains duplicate column names.")

        records = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                raise ValueError(
                    f"CSV line {reader.line_num} has {len(row)} fields; expected {len(columns)}."
                )
            records.append(row)
    except UnicodeDecodeError as error:
        raise ValueError("Save the CSV with UTF-8 encoding and upload it again.") from error
    except csv.Error as error:
        raise ValueError(f"Invalid CSV formatting: {error}.") from error

    if not records:
        raise ValueError("The CSV has a header but no sensor records.")
    return pd.DataFrame(records, columns=columns, dtype=object)
