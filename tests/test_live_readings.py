from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from data_processing.live_readings import ESTIMATED, PENDING, elapsed_time, estimate_readings, save_reading
from visualization.graphs import sensor_figure


def enter(*rows, date_time=False):
    readings = ()
    for time, value in rows:
        readings = save_reading(readings, time, value, date_time)
    return readings


@pytest.mark.parametrize("method", ["Linear", "Lagrange"])
def test_consecutive_gaps_and_boundaries_use_original_anchors(method):
    readings = enter(("10", None), ("5", "35"), ("0", None), ("1", "15"), ("2", None), ("3", None))
    original, result, table = estimate_readings(readings, method)

    assert table["Timestamp"].tolist() == ["0", "1", "2", "3", "5", "10"]
    np.testing.assert_allclose(original, [np.nan, 15, np.nan, np.nan, 35, np.nan])
    np.testing.assert_allclose(result.values, [np.nan, 15, 20, 25, 35, np.nan])
    assert table["Status"].tolist() == [PENDING, "Recorded", ESTIMATED, ESTIMATED, "Recorded", PENDING]


def test_datetime_offsets_elapsed_hours_and_duplicate_detection():
    readings = enter(
        ("2026-01-01T04:00+00:00", "12"),
        ("2026-01-01T02:30+01:00", None),
        ("2026-01-01T00:00", "4"),
        date_time=True,
    )
    np.testing.assert_allclose(elapsed_time(readings), [0, 1.5, 4])
    assert estimate_readings(readings, "Linear")[2]["Estimated Reading"].iloc[1] == 7
    with pytest.raises(ValueError, match="already has an entry"):
        save_reading(readings, "2026-01-01T05:00+01:00", "17", True)


@pytest.mark.parametrize(("timestamp", "value", "date_time", "message"), [
    ("nan", "37", False, "Time must be a finite number"),
    ("1", "Infinity", False, "Sensor reading must be a finite number"),
    ("1", "word", False, "Sensor reading must be a finite number"),
    ("1", "", False, "Sensor reading must be a finite number"),
    ("tomorrow", "15", True, "Enter an ISO date-time"),
])
def test_invalid_input_is_rejected(timestamp, value, date_time, message):
    with pytest.raises(ValueError, match=message):
        save_reading((), timestamp, value, date_time)


def test_edit_retains_other_entries_and_validates_collisions():
    readings = enter(("1", "10"), ("2", None), ("4", "40"))
    with pytest.raises(ValueError, match="already has an entry"):
        save_reading(readings, "1.0", "22", False, replace=readings[1])
    with pytest.raises(ValueError, match="already has an entry"):
        save_reading(readings, "4.00", None, False)

    updated = save_reading(readings, "3", "31", False, replace=readings[1])
    assert updated[0] is readings[0]
    assert updated[2] is readings[2]
    assert updated[1].value == 31
    missing = save_reading(updated, "3", None, False, replace=updated[1])
    assert estimate_readings(missing, "Linear")[2]["Estimated Reading"].iloc[1] == 30


def test_live_graph_keeps_boundary_gaps_visible():
    readings = enter(("1", None), ("2", "1"), ("3", None), ("5", "7"), ("10", None))
    original, result, _ = estimate_readings(readings, "Linear")
    times = np.array([entry.time for entry in readings])
    figure = sensor_figure(times, original, "Temperature", {"Linear": result}, numeric_axis="Time", show_all_gaps=True)
    axis = figure.axes[0]
    left, right = axis.get_xlim()
    assert left < 1 < 10 < right
    assert axis.get_xlabel() == "Time"
    np.testing.assert_array_equal(axis.collections[-1].get_offsets()[:, 0], [1, 3, 10])


def test_streamlit_live_workflow_and_dataset_recovery():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    assert app.tabs[0].metric[2].value == "1,147"

    def live_tab():
        return next(tab for tab in app.tabs if tab.label == "Live Sensor Reading Estimator")

    def submit(time, value, button_label):
        live = live_tab()
        next(field for field in live.text_input if field.label == "Time").set_value(time)
        next(field for field in live.text_input if field.label == "Sensor reading").set_value(value)
        next(button for button in live.button if button.label == button_label).click().run()
        assert not app.exception

    submit("1", "1", "Add Reading")
    submit("2", "4", "Add Reading")
    submit("3", "", "Mark Reading as Missing")
    assert PENDING in live_tab().dataframe[0].value["Status"].tolist()
    submit("4", "16", "Add Reading")
    live = live_tab()
    assert live.dataframe[0].value["Estimated Reading"].iloc[2] == 10
    assert len(live.get("image")) == 1
    next(widget for widget in live.selectbox if widget.label == "Live interpolation method").select("Lagrange").run()
    assert live_tab().dataframe[0].value["Estimated Reading"].iloc[2] == pytest.approx(9)
    assert live_tab().get("download_button")

    next(widget for widget in live_tab().selectbox if widget.label == "Reading to edit").select("3").run()
    next(field for field in live_tab().text_input if field.label == "Corrected time").set_value("3.5")
    next(button for button in live_tab().button if button.label == "Save changes").click().run()
    edited = live_tab().dataframe[0].value
    assert edited["Timestamp"].iloc[2] == "3.5"
    assert edited["Estimated Reading"].iloc[2] == pytest.approx(12.25)

    next(widget for widget in live_tab().selectbox if widget.label == "Live sensor parameter").select("Pressure").run()
    assert not live_tab().dataframe
    submit("1", "50", "Add Reading")
    next(widget for widget in live_tab().selectbox if widget.label == "Live sensor parameter").select("Temperature").run()
    assert live_tab().dataframe[0].value["Estimated Reading"].iloc[2] == pytest.approx(12.25)

    next(widget for widget in live_tab().selectbox if widget.label == "Reading to edit").select("2").run()
    next(button for button in live_tab().button if button.label == "Remove reading").click().run()
    assert live_tab().dataframe[0].value["Timestamp"].tolist() == ["1", "3.5", "4"]
    assert any("first-degree Lagrange" in caption.value for caption in live_tab().caption)

    next(button for button in live_tab().button if button.label == "Reset Readings").click().run()
    assert not live_tab().dataframe
    assert app.session_state["live_readings"] == {}
    assert not app.exception
