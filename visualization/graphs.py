"""Matplotlib plots of the selected sensor in chronological order."""

import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from data_processing.missing_values import RECOVERED, Recovery

COLORS = {"Linear": "#07897A", "Lagrange": "#A35B22"}
MARKERS = {"Linear": "o", "Lagrange": "^"}


def sensor_figure(
    timestamps: pd.Series | np.ndarray,
    readings: np.ndarray,
    label: str,
    results: dict[str, Recovery] | None = None,
    *,
    numeric_axis: str | None = None,
    show_all_gaps: bool = False,
) -> Figure:
    figure = Figure(figsize=(10.5, 3.2), layout="constrained", facecolor="white")
    ax = figure.subplots()
    known = np.isfinite(readings)
    series = [readings, *(result.values for result in (results or {}).values())]
    peak = max(np.max(np.abs(values), where=np.isfinite(values), initial=0.0) for values in series)
    scale = 1e200 if peak > 1e200 else 1.0
    if numeric_axis is not None and np.abs(timestamps).max() > 1e200:
        timestamps = timestamps / 1e200
        numeric_axis += " (×10²⁰⁰)"
    ax.plot(timestamps, readings / scale, color="#344F5D", linewidth=0.8, alpha=0.65)
    ax.scatter(timestamps[known], readings[known] / scale, color="#344F5D", s=9, label="Recorded")
    if results is None:
        ax.set_title(f"{label} · Original readings", loc="left", fontweight="bold")
    else:
        title = "Method comparison" if len(results) == 2 else "Reconstructed readings"
        ax.set_title(f"{label} · {title}", loc="left", fontweight="bold")
        for method, result in results.items():
            recovered = result.status == RECOVERED
            ax.scatter(
                timestamps[recovered], result.values[recovered] / scale, marker=MARKERS[method],
                color=COLORS[method], s=32, edgecolor="white", linewidth=0.35,
                label=f"{method} estimates", zorder=3,
            )

    missing = ~known if results is None or show_all_gaps else np.logical_and.reduce(
        [~np.isfinite(result.values) for result in results.values()]
    )
    if missing.any():
        missing_times = timestamps[missing]
        ax.scatter(
            missing_times, np.full(missing.sum(), 0.025), transform=ax.get_xaxis_transform(),
            color="#B45539", marker="|", s=110, label="Missing · position only", zorder=4,
        )
        # Axis-relative gap markers belong in the time limits even beyond the recorded range.
        ax.update_datalim(np.column_stack((ax.convert_xunits(missing_times), np.zeros(missing.sum()))), updatey=False)
        ax.autoscale_view(scaley=False)
    if numeric_axis is None:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set(xlabel=numeric_axis or "Timestamp (UTC)", ylabel=f"{label} (×10²⁰⁰)" if scale != 1 else label)
    ax.grid(axis="y", color="#E1ECE8")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#A4BAB3")
    ax.tick_params(colors="#455E59", labelsize=8)
    ax.legend(loc="best", facecolor="#F3F8F6", edgecolor="#D9E4E1", fontsize=8)
    return figure
