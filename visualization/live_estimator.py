"""Streamlit workspace for entering, correcting, and estimating sensor readings."""

from datetime import datetime

import numpy as np
import streamlit as st

from data_processing.live_readings import PENDING, UNABLE, Reading, elapsed_time, estimate_readings, save_reading
from data_processing.missing_values import METHODS, RECOVERED
from visualization.dashboard import download_csv
from visualization.graphs import sensor_figure

SENSORS = (
    "Temperature", "Pressure", "Humidity", "Vibration", "Voltage", "Rotation Speed", "Current",
    "Oil Level", "Load", "Motor Temperature", "Gearbox Temperature", "Sound Level", "Fan Speed",
    "Reactive Power", "Active Power",
)
NUMERIC_TIME = "Numerical time"
DATE_TIME = "Date and time"


def show_live_estimator() -> None:
    st.header("Live Sensor Reading Estimator")
    st.caption("Enter recorded values or mark gaps as missing. Estimates update when a gap has a recorded value on both sides.")
    with st.container(border=True, key="live_controls"):
        sensor_column, method_column = st.columns(2)
        sensor = sensor_column.selectbox("Live sensor parameter", SENSORS)
        method = method_column.selectbox(
            "Live interpolation method", METHODS, format_func=lambda name: f"{name} Interpolation",
        )

        series = st.session_state.setdefault("live_readings", {})
        readings = series.get(sensor, ())
        date_time = bool(readings and isinstance(readings[0].time, datetime))
        time_control = st.empty()
        if readings:
            time_control.caption(f"Time format · {DATE_TIME if date_time else NUMERIC_TIME}")
        else:
            date_time = time_control.radio(
                "Time format", (NUMERIC_TIME, DATE_TIME), horizontal=True,
                key=f"live_time_format_{sensor}",
            ) == DATE_TIME

        time_hint = "e.g. 2026-11-21 14:30" if date_time else "e.g. 3.5"
        time_help = "ISO date-time. Times without a timezone offset use UTC." if date_time else None
        with st.form("live_add", clear_on_submit=True):
            time_column, value_column = st.columns(2)
            timestamp = time_column.text_input("Time", placeholder=time_hint, help=time_help, max_chars=64)
            value = value_column.text_input("Sensor reading", placeholder="e.g. 27.5", max_chars=64)
            add_column, gap_column = st.columns(2)
            add = add_column.form_submit_button("Add Reading", type="primary")
            mark = gap_column.form_submit_button("Mark Reading as Missing")
        if add or mark:
            try:
                readings = save_reading(readings, timestamp, value if add else None, date_time)
                series[sensor] = readings
                time_control.caption(f"Time format · {DATE_TIME if date_time else NUMERIC_TIME}")
            except ValueError as error:
                st.error(str(error))

    if readings:
        edit_reading(series, sensor, readings, date_time)

    calculate_column, reset_column = st.columns(2)
    calculate = calculate_column.button("Calculate Missing Values", type="primary", disabled=not readings)
    if reset_column.button("Reset Readings", help="Clear all manually entered sensors", disabled=not series):
        series.clear()
        st.rerun()
    if not readings:
        st.info(f"Add a {sensor.lower()} reading or mark a timestamp as missing to start.")
        return

    show_live_results(readings, sensor, method, date_time, calculate)


def edit_reading(
    series: dict[str, tuple[Reading, ...]], sensor: str, readings: tuple[Reading, ...], date_time: bool,
) -> None:
    with st.expander("Edit readings"):
        by_timestamp = {entry.timestamp: entry for entry in readings}
        selected = st.selectbox("Reading to edit", list(by_timestamp), key=f"live_edit_select_{sensor}")
        entry = by_timestamp[selected]
        with st.form(f"live_edit_{sensor}_{selected}"):
            time_column, value_column = st.columns(2)
            new_time = time_column.text_input("Corrected time", value=entry.timestamp, max_chars=64)
            new_value = value_column.text_input(
                "Corrected sensor reading", value="" if entry.value is None else str(entry.value), max_chars=64,
            )
            missing = st.checkbox("Mark as missing", value=entry.value is None)
            save_column, remove_column = st.columns(2)
            save = save_column.form_submit_button("Save changes", type="primary")
            remove = remove_column.form_submit_button("Remove reading")
        if save or remove:
            try:
                updated = (
                    save_reading(readings, new_time, None if missing else new_value, date_time, replace=entry)
                    if save else tuple(reading for reading in readings if reading != entry)
                )
                if updated:
                    series[sensor] = updated
                else:
                    series.pop(sensor)
                st.rerun()
            except ValueError as error:
                st.error(str(error))


def show_live_results(
    readings: tuple[Reading, ...], sensor: str, method: str, date_time: bool, calculate: bool,
) -> None:
    original, result, table = estimate_readings(readings, method)
    recovered = result.status == RECOVERED
    count = int(recovered.sum())
    pending = int((table["Status"] == PENDING).sum())
    unable = int((table["Status"] == UNABLE).sum())
    with st.container(border=True, key="live_results"):
        st.subheader(f"{sensor} readings")
        if count == 1:
            st.success(f"Estimated {sensor}: {result.values[recovered][0]:.10g} · {method} Interpolation")
        elif count:
            st.success(f"{count:,} {sensor.lower()} readings estimated with {method} Interpolation.")
        if pending:
            st.info(f"Pending missing readings: {pending:,}. Add recorded values on both sides of each gap.")
        if unable:
            st.warning(f"{unable:,} missing readings could not be estimated because the result was numerically invalid.")
        if calculate and not (count or pending or unable):
            st.info("There are no missing readings to calculate.")
        if method == "Lagrange" and count and np.isfinite(original).sum() == 2:
            st.caption("With two recorded points, first-degree Lagrange gives the same estimate as Linear Interpolation.")
        display = table.assign(**{
            "Original Reading": table["Original Reading"].map(
                lambda reading: f"{reading:.10g}" if np.isfinite(reading) else "Missing"
            ),
        })
        st.dataframe(
            display, hide_index=True,
            column_config={"Estimated Reading": st.column_config.NumberColumn(format="%.10g")},
        )

    times = elapsed_time(readings) if date_time else np.array([reading.time for reading in readings])
    axis_label = "Time" if not date_time else f"Elapsed hours from {readings[0].timestamp}"
    with st.container(border=True, key="live_chart"):
        st.pyplot(sensor_figure(
            times, original, sensor, {method: result} if count else None,
            numeric_axis=axis_label, show_all_gaps=True,
        ))
    download_csv("Download live readings CSV", table, f"live_{sensor.lower().replace(' ', '_')}_{method.lower()}.csv")
