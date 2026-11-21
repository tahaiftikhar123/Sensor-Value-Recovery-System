from math import sqrt

import numpy as np
import pandas as pd
import pytest

from evaluation import accuracy


def test_error_formulas():
    mae, rmse = accuracy.error_metrics(np.array([1, 4, 8]), np.array([2, 1, 10]))
    assert mae == 2
    assert rmse == pytest.approx(sqrt(14 / 3))


def test_common_reproducible_targets_and_no_leakage(monkeypatch):
    hours = np.arange(30, dtype=float)
    readings = hours ** 2 + 4
    observed_inputs = []
    reconstruct = accuracy.reconstruct

    def inspect_visible(times, values, method):
        observed_inputs.append(values.copy())
        return reconstruct(times, values, method)

    monkeypatch.setattr(accuracy, "reconstruct", inspect_visible)
    first = accuracy.evaluate(hours, readings)
    second = accuracy.evaluate(hours, readings)
    assert first.held_out == first.compared == 3
    pd.testing.assert_frame_equal(first.metrics, second.metrics)
    hidden = np.isnan(observed_inputs[0])
    assert hidden.sum() == 3
    assert not hidden[0] and not hidden[-1]
    assert np.isfinite(readings).all()
    for visible in observed_inputs:
        np.testing.assert_array_equal(np.isnan(visible), hidden)
        np.testing.assert_array_equal(visible[~hidden], readings[~hidden])
    assert first.metrics.loc[0, "MAE"] > 0
    assert first.metrics.loc[1, "RMSE"] == pytest.approx(0, abs=1e-12)


def test_evaluation_requires_a_bracketed_known_target():
    result = accuracy.evaluate(np.arange(3), np.array([2.0, np.nan, 4.0]))
    assert result.held_out == result.compared == 0
    assert result.metrics[["MAE", "RMSE"]].isna().all().all()
