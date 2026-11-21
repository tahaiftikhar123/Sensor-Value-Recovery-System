"""Streamlit entry point for the Sensor Data Recovery System."""

from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from data_processing.data_loader import SAMPLE_PATH, load_csv
from data_processing.missing_values import (
    INVALID,
    METHODS,
    RECOVERED,
    missing_summary,
    reconstruct,
    recover_all,
    sensor_label,
    sensor_results,
)
from data_processing.validation import METADATA_NAMES, TIME_NAMES, normalized, validate_data
from evaluation.accuracy import evaluate
from visualization.dashboard import dashboard_header, download_csv, load_dashboard_style
from visualization.graphs import sensor_figure
from visualization.live_estimator import show_live_estimator

COMPARE = "Compare both methods"


@st.cache_data(show_spinner=False, max_entries=5)
def read_dataset(content: bytes) -> pd.DataFrame:
    return load_csv(content)


def show_evaluation(hours, readings) -> None:
    evaluation = evaluate(hours, readings)
    if not evaluation.held_out:
        st.info("Accuracy evaluation needs at least three recorded readings.")
        return

    st.caption(
        f"{evaluation.held_out:,} recorded readings were temporarily hidden. "
        f"Both methods were scored on the same {evaluation.compared:,} readings with finite estimates."
    )
    st.dataframe(
        evaluation.metrics, hide_index=True,
        column_config={metric: st.column_config.NumberColumn(format="%.4g") for metric in ("MAE", "RMSE")},
    )
    if evaluation.compared < evaluation.held_out:
        st.warning("Some held-out readings had an invalid estimate and were excluded from both scores.")

    scores = evaluation.metrics.set_index("Method")
    for metric in ("MAE", "RMSE"):
        values = scores[metric]
        if values.isna().any():
            st.caption(f"{metric} comparison is unavailable.")
        elif values.nunique() == 1:
            st.write(f"Both methods tied on {metric} for the evaluated readings.")
        else:
            st.write(f"{values.idxmin()} had the lower {metric} on the evaluated readings.")


def show_dataset_recovery() -> None:
    hero = st.empty()

    with st.sidebar:
        st.html(
            '<div class="recovery-controls">'
            '<span class="control-mark" aria-hidden="true"><i></i><i></i><i></i></span>'
            'Dataset recovery controls</div>'
        )
    source = st.sidebar.radio("Dataset", ("Sample IIoT dataset", "Upload a CSV"))
    try:
        if source == "Upload a CSV":
            upload = st.sidebar.file_uploader("Sensor CSV", type=["csv"])
            if upload is None:
                hero.html(dashboard_header())
                st.info("Upload a CSV with a timestamp column and numerical sensor readings.")
                return
            content, filename = upload.getvalue(), upload.name
        else:
            content, filename = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
        original = read_dataset(content)
        columns = original.columns.tolist()
        time_options = [name for name in columns if normalized(name) not in METADATA_NAMES]
        if not time_options:
            raise ValueError("The CSV needs a timestamp column and at least one sensor column.")
        default_time = next(
            (index for index, name in enumerate(time_options) if normalized(name) == "timestamp"),
            next((index for index, name in enumerate(time_options) if normalized(name) in TIME_NAMES), None),
        )
        dataset_id = sha256(content).hexdigest()
        time_column = st.sidebar.selectbox(
            "Time column", time_options, index=default_time, key=f"time_{dataset_id}",
            placeholder="Choose a timestamp column",
        )
        if time_column is None:
            hero.html(dashboard_header())
            st.info("Choose the column that contains the measurement timestamps.")
            return
        excluded = ()
        if source == "Upload a CSV":
            candidates = [
                name for name in time_options if name != time_column and normalized(name) not in TIME_NAMES
            ]
            with st.sidebar.expander("Non-sensor columns"):
                excluded = tuple(st.multiselect(
                    "Exclude IDs, categories, or other metadata", candidates,
                    key=f"exclude_{dataset_id}_{time_column}",
                ))
        data = validate_data(original, time_column, excluded)
    except FileNotFoundError:
        hero.html(dashboard_header())
        st.error(f"The included dataset is missing. Restore {SAMPLE_PATH.name} in the datasets folder or upload a CSV.")
        return
    except (ValueError, OverflowError, OSError) as error:
        hero.html(dashboard_header())
        st.error(f"Cannot use this dataset. {error}")
        return

    readings = data.readings
    dataset_key = (dataset_id, time_column, excluded)
    sensor = st.sidebar.selectbox(
        "Sensor", readings.columns.tolist(), format_func=sensor_label, key=f"sensor_{dataset_key}",
    )
    method_titles = {"Linear": "Linear interpolation", "Lagrange": "Lagrange interpolation", COMPARE: COMPARE}
    mode = st.sidebar.radio("Interpolation method", (*METHODS, COMPARE), format_func=method_titles.get)
    label = sensor_label(sensor)
    selected = readings[sensor]
    missing = selected.isna()
    values = selected.to_numpy()
    missing_count = int(missing.sum())
    hero.html(dashboard_header(label, values))

    st.subheader("Dataset overview")
    st.caption(filename)
    total_missing = int(readings.isna().sum().sum())
    with st.container(key="dataset_stats"):
        stat = st.columns(4)
        stat[0].metric("Records", f"{len(original):,}")
        stat[1].metric("Sensor parameters", len(readings.columns))
        stat[2].metric("Missing readings", f"{total_missing:,}")
        stat[3].metric("Missing share", f"{100 * total_missing / readings.size:.2f}%")
    if data.warnings:
        with st.expander("Dataset validation messages", expanded=True):
            for warning in data.warnings:
                st.warning(warning)
    with st.expander("Dataset preview"):
        st.dataframe(original.head(20), hide_index=True)
        st.caption(f"Showing the first {min(len(original), 20):,} records in their original order.")

    analysis_tab, recovery_tab, accuracy_tab, export_tab = st.tabs(
        ("Missing values", "Interpolation & graphs", "Accuracy evaluation", "CSV export")
    )
    with analysis_tab:
        with st.container(key="analysis_layout"):
            summary_column, detail_column = st.columns([1.55, 1], gap="medium")
        with summary_column, st.container(border=True, key="gap_summary"):
            st.subheader("Missing readings by sensor")
            st.dataframe(
                missing_summary(readings), hide_index=True, height=350,
                column_config={
                    "Missing %": st.column_config.ProgressColumn(
                        "Missing %", format="%.2f%%", min_value=0, max_value=100,
                    ),
                },
            )
        with detail_column, st.container(border=True, key="sensor_detail"):
            st.subheader(label)
            col1, col2, col3 = st.columns(3)
            col1.metric("Available", f"{len(selected) - missing_count:,}")
            col2.metric("Missing", f"{missing_count:,}")
            col3.metric("Missing %", f"{100 * missing_count / len(selected):.2f}%")
            with st.expander("Timestamps of missing readings"):
                if missing_count:
                    st.dataframe(
                        original.loc[selected.index[missing.to_numpy()], [time_column]],
                        hide_index=True, height=250,
                    )
                else:
                    st.write("All readings are available.")
            with st.expander("Original sensor readings"):
                st.dataframe(
                    original[[time_column, sensor]], hide_index=True, height=300,
                )

    result_key = (dataset_key, sensor, mode)
    with recovery_tab:
        heading, action = st.columns([3, 1.5], vertical_alignment="center")
        heading.subheader(f"{label} recovery")
        if action.button("Apply interpolation", icon=":material/auto_fix_high:", type="primary", width="stretch"):
            with st.spinner("Estimating missing readings…"):
                methods = METHODS if mode == COMPARE else (mode,)
                st.session_state.single_recovery = (
                    result_key, {method: reconstruct(data.hours, values, method) for method in methods}
                )
        previous_key, results = st.session_state.get("single_recovery", (None, {}))
        results = results if previous_key == result_key else {}
        result_table = None
        if results:
            result_table = sensor_results(original, data, sensor, results)
            with st.container(key="recovery_results"):
                for method, result in results.items():
                    recovered = int((result.status == RECOVERED).sum())
                    st.write(f"**{method}** · {recovered:,} recovered · {missing_count - recovered:,} still missing")
                    reasons = pd.Series(result.status[missing.to_numpy()]).value_counts()
                    for status, count in reasons.items():
                        if status == INVALID:
                            st.warning(f"{method}: {count:,} estimates were numerically invalid.")
                        elif status != RECOVERED:
                            st.caption(f"{method}: {status.lower()} ({count:,}).")
            if missing_count:
                display = result_table.loc[selected.index[missing.to_numpy()]].copy()
                display["Original"] = "Missing"
                for method in results:
                    display[method] = pd.to_numeric(display[method], errors="coerce")
                st.dataframe(display, hide_index=True, height=240, column_config={
                    name: st.column_config.NumberColumn(format="%.6g")
                    for name in (*results, "Lagrange − Linear") if name in display
                })
            else:
                st.info("All readings for this sensor are available.")
        else:
            st.caption("Apply the selected method to estimate the missing readings and enable a CSV download.")

        with st.container(border=True, key="recovery_charts"):
            if results:
                st.pyplot(sensor_figure(data.timestamps, values, label, results))
            st.pyplot(sensor_figure(data.timestamps, values, label))

    with accuracy_tab, st.container(border=True, key="accuracy_card"):
        st.subheader(f"{label} accuracy evaluation")
        show_evaluation(data.hours, values)

    with export_tab, st.container(border=True, key="sensor_export"):
        st.subheader(f"{label} CSV export")
        stem = Path(filename).stem
        if result_table is not None:
            method_slug = "linear_vs_lagrange" if mode == COMPARE else mode.lower()
            download_csv("Download sensor CSV", result_table, f"{stem}_{normalized(sensor)}_{method_slug}.csv")
        else:
            st.info("Apply interpolation for this sensor to create its download.")

    with export_tab, st.container(border=True, key="batch_export"):
        st.subheader("Recover all sensors")
        all_method = mode if mode in METHODS else st.radio(
            "All-sensor interpolation method", METHODS, horizontal=True, format_func=method_titles.get,
        )
        all_key = (dataset_key, all_method)
        if st.button("Recover all sensors", type="primary"):
            with st.spinner("Recovering all sensor parameters…"):
                st.session_state.all_recovery = (all_key, recover_all(original, data, all_method))
        previous_all_key, all_export = st.session_state.get("all_recovery", (None, None))
        if previous_all_key == all_key and all_export is not None:
            completed, summary = all_export
            remaining = int(summary["Remaining Missing"].sum())
            recovered = int(summary["Recovered"].sum())
            st.success(f"{all_method}: {recovered:,} readings recovered across {len(summary):,} sensors.")
            if remaining:
                st.info(f"{remaining:,} readings remain missing.")
            if summary["Numerically Invalid"].sum():
                st.warning("Numerically invalid estimates are counted in the recovery summary.")
            st.dataframe(summary, hide_index=True)
            download_csv("Download all-sensor CSV", completed, f"{stem}_all_sensors_{all_method.lower()}.csv")


def main() -> None:
    st.set_page_config(page_title="Sensor Data Recovery System", layout="wide")
    load_dashboard_style()
    dataset_tab, live_tab = st.tabs(("Dataset Recovery", "Live Sensor Reading Estimator"))
    with dataset_tab:
        show_dataset_recovery()
    with live_tab:
        show_live_estimator()


if __name__ == "__main__":
    main()
