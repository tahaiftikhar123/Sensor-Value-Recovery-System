import numpy as np
import pytest

from data_processing.missing_values import INVALID, reconstruct
from numerical_methods.lagrange_interpolation import lagrange_interpolate, local_indices


@pytest.mark.parametrize(("xs", "polynomial"), [
    ([0, 3], lambda x: 4 * x - 3),
    ([0, 1, 4], lambda x: x ** 2 - 3 * x + 7),
    ([0, 1, 3, 6], lambda x: 2 * x ** 3 - x + 1),
])
def test_polynomial_interpolation(xs, polynomial):
    xs = np.array(xs, dtype=float)
    assert lagrange_interpolate(2.0, xs, polynomial(xs)) == pytest.approx(polynomial(2.0))


def test_local_selection_retains_the_bracketing_pair():
    hours = np.array([0, 1, 2, 3, 10, 25, 50])
    np.testing.assert_array_equal(local_indices(hours, 4, 4), [1, 2, 3, 4])


def test_consecutive_gaps_use_original_anchors_for_a_cubic():
    hours = np.array([0.0, 1.0, 2.0, 3.0, 6.0, 8.0])
    expected = hours ** 3 - 4 * hours
    readings = expected.copy()
    readings[[2, 3]] = np.nan
    result = reconstruct(hours, readings, "Lagrange")
    np.testing.assert_allclose(result.values, expected, atol=1e-12)
    np.testing.assert_array_equal(result.values[[0, 1, 4, 5]], readings[[0, 1, 4, 5]])


def test_duplicate_timestamps_fail_clearly():
    with pytest.raises(ValueError, match="distinct"):
        lagrange_interpolate(1.0, np.array([0, 3, 3]), np.array([0, 9, 9]))
    with pytest.raises(ValueError, match="distinct"):
        reconstruct(np.array([0, 3, 3]), np.array([0, np.nan, 9]), "Lagrange")


def test_numerically_invalid_result_is_not_replaced_with_linear():
    hours = np.array([0, 1, 50, 100, 101])
    values = np.array([1e308, -1e308, np.nan, 1e308, -1e308])
    result = reconstruct(hours, values, "Lagrange")
    assert np.isnan(result.values[2])
    assert result.status[2] == INVALID
    assert np.isfinite(reconstruct(hours, values, "Linear").values[2])
