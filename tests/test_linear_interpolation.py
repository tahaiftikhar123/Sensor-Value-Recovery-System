import numpy as np
import pytest

from data_processing.missing_values import BOUNDARY, INSUFFICIENT, RECORDED, RECOVERED, reconstruct
from numerical_methods.linear_interpolation import linear_interpolate


def test_two_bracketing_readings():
    assert linear_interpolate(12, 11, 27, 13, 31) == 29
    with pytest.raises(ValueError, match="inside"):
        linear_interpolate(14, 11, 27, 13, 31)


def test_consecutive_gaps_use_elapsed_time_and_preserve_values():
    hours = np.array([0.0, 1.0, 3.0, 8.0, 10.0, 11.0])
    values = np.array([2.0, np.nan, np.nan, 18.0, np.nan, 24.0])
    result = reconstruct(hours, values, "Linear")
    np.testing.assert_allclose(result.values, [2, 4, 8, 18, 22, 24])
    assert list(result.status) == [RECORDED, RECOVERED, RECOVERED, RECORDED, RECOVERED, RECORDED]
    assert np.isnan(values[[1, 2, 4]]).all()


@pytest.mark.parametrize("method", ["Linear", "Lagrange"])
def test_boundaries_are_not_extrapolated(method):
    result = reconstruct(np.arange(5), np.array([np.nan, 2, np.nan, 4, np.nan]), method)
    np.testing.assert_allclose(result.values, [np.nan, 2, 3, 4, np.nan])
    assert list(result.status[[0, 4]]) == [BOUNDARY, BOUNDARY]


@pytest.mark.parametrize("values", [[np.nan, np.nan, np.nan], [np.nan, 8, np.nan]])
def test_insufficient_observations(values):
    values = np.array(values)
    result = reconstruct(np.arange(3), values, "Linear")
    np.testing.assert_array_equal(result.values, values)
    assert set(result.status[np.isnan(values)]) == {INSUFFICIENT}
