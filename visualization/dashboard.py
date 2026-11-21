"""Render the dashboard header and its preview of the selected sensor."""

from html import escape
from pathlib import Path

import numpy as np
import streamlit as st

STYLE = Path(__file__).with_name("dashboard.css")


def load_dashboard_style() -> None:
    st.html(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")


def dashboard_header(label: str = "", readings: np.ndarray | None = None) -> str:
    signal = ""
    if readings is not None:
        sample = readings[:72]
        finite = np.isfinite(sample)
        heights = np.full(len(sample), 55.0)
        if finite.any():
            recorded = sample[finite]
            scaled = recorded / max(float(np.abs(recorded).max()), 1e-300)
            low, high = scaled.min(), scaled.max()
            if high > low:
                heights[finite] = 18 + 80 * (scaled - low) / (high - low)

        bars = "".join(
            f'<span class="{"signal-mark" if valid else "signal-gap"}" '
            f'style="--level:{height:.1f}%;--lag:{index * 7}ms"></span>'
            for index, (valid, height) in enumerate(zip(finite, heights))
        )
        safe_label = escape(label)
        signal = f"""
            <div class="signal-preview" role="img"
                 aria-label="{safe_label}, first {len(sample)} readings. {int((~finite).sum())} missing.">
                <div class="signal-heading" aria-hidden="true">
                    <span>{safe_label}</span><span>FIRST {len(sample)} READINGS</span>
                </div>
                <div class="signal-bars" aria-hidden="true">{bars}</div>
                <div class="signal-legend" aria-hidden="true"><span>Recorded</span><span>Missing</span></div>
            </div>
        """

    return f"""
        <section class="dashboard-hero">
            <div class="hero-copy">
                <p class="hero-context">IIoT / NUMERICAL ANALYSIS</p>
                <h1>Sensor Data<br class="hero-break"> Recovery System</h1>
                <p class="hero-description">Missing Sensor Reading Estimation Using Numerical Interpolation</p>
            </div>
            {signal}
        </section>
    """
