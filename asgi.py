"""Expose the Streamlit dashboard to Vercel's Python runtime."""

from pathlib import Path

import streamlit as st

app = st.App(Path(__file__).with_name("app.py"))
