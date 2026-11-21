import numpy as np
import pandas as pd
import pytest

from data_processing.data_loader import SAMPLE_PATH, load_csv
from data_processing.missing_values import METHODS, missing_summary, reconstruct, recover_all, sensor_results
from data_processing.validation import validate_data


def test_validation_sorts_tracks_incomplete_sensors_and_preserves_text():
    original = load_csv(
        b"logged_at,pressure,empty,voltage,machine_failure,code,note\r\n"
        b"2024-01-01 04:00,8.000,,inf,1,003,down\r\n"
        b"2024-01-01 00:00,0.00,,,0,001,up\r\n"
        b"2024-01-01 01:30,bad,,12,0,002,up\r\n"
    )
    data = validate_data(original, "logged_at", ("code",))
    np.testing.assert_allclose(data.hours, [0, 1.5, 4])
    assert list(data.readings) == ["pressure", "empty", "voltage"]
    assert missing_summary(data.readings)["Missing"].tolist() == [1, 3, 2]
    assert any("invalid" in warning for warning in data.warnings)
    assert any("sorted" in warning for warning in data.warnings)
    assert any("at least two" in warning for warning in data.warnings)
    assert any("no valid readings" in warning for warning in data.warnings)
    recovered, _ = recover_all(original, data, "Linear")
    assert recovered["pressure"].tolist() == ["8.000", "0.00", 3.0]
    assert original["pressure"].tolist() == ["8.000", "0.00", "bad"]
    pd.testing.assert_series_equal(recovered["logged_at"], original["logged_at"])
    pd.testing.assert_series_equal(recovered["code"], original["code"])


@pytest.mark.parametrize("raw", [
    b"", b"timestamp,sensor", b"timestamp,sensor\n2024-01-01,5,unexpected",
    b'timestamp,sensor\n2024-01-01,"unterminated', b"timestamp,timestamp\n2024-01-01,3",
    b"timestamp,\n2024-01-01,3", b"\xff",
])
def test_invalid_csv(raw):
    with pytest.raises(ValueError):
        load_csv(raw)


@pytest.mark.parametrize(("raw", "message"), [
    (b"date,voltage\n2024-01-01,1", "Select a timestamp column"),
    (b"timestamp,voltage\ninvalid,1", "invalid or missing timestamp"),
    (b"timestamp,voltage\n,1", "invalid or missing timestamp"),
    (b"timestamp,voltage\n2024-01-01T00:00Z,1\n2024-01-01T01:00+01:00,2", "duplicate timestamps"),
    (b"timestamp,note\n2024-01-01,ok", "No numerical sensor columns"),
])
def test_invalid_dataset(raw, message):
    with pytest.raises(ValueError, match=message):
        validate_data(load_csv(raw), "timestamp")


@pytest.mark.parametrize("method", METHODS)
def test_sample_all_sensors_and_csv_round_trip(method):
    original = load_csv(SAMPLE_PATH.read_bytes())
    data = validate_data(original, "timestamp")
    assert original.shape == (1000, 17)
    assert data.readings.isna().sum().tolist() == [
        76, 91, 78, 85, 67, 65, 72, 91, 77, 76, 71, 81, 66, 82, 69,
    ]

    completed, summary = recover_all(original, data, method)
    exported = load_csv(completed.to_csv(index=False, encoding="utf-8").encode())
    assert exported.shape == original.shape
    assert exported.columns.tolist() == original.columns.tolist()
    assert summary["Recovered"].sum() == 1147
    assert summary["Remaining Missing"].sum() == 0
    for column in original:
        known = original[column] != ""
        pd.testing.assert_series_equal(exported.loc[known, column], original.loc[known, column])
        assert (exported[column] != "").all()


def test_single_and_comparison_exports_keep_original_order_and_statuses():
    original = load_csv(
        b"date,temperature\n2024-01-01 02:00,\n2024-01-01 00:00,02.00\n"
        b"2024-01-01 08:00,10\n2024-01-02 00:00,\n"
    )
    data = validate_data(original, "date")
    readings = data.readings["temperature"].to_numpy()
    comparison = {method: reconstruct(data.hours, readings, method) for method in METHODS}
    for selected in [{"Linear": comparison["Linear"]}, comparison]:
        table = sensor_results(original, data, "temperature", selected)
        exported = load_csv(table.to_csv(index=False).encode())
        assert exported["Timestamp"].tolist() == original["date"].tolist()
        assert exported["Original"].tolist() == ["", "02.00", "10", ""]
        for method in selected:
            assert exported[method].tolist() == ["4.0", "02.00", "10", ""]
            assert exported[f"{method} status"].tolist() == [
                "Recovered", "Recorded", "Recorded", "Outside known time range",
            ]
        assert ("Lagrange − Linear" in exported) == (len(selected) == 2)
