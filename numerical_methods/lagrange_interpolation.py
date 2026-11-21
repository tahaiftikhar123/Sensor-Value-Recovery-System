"""Evaluate a local polynomial using the Lagrange basis."""

from math import fsum

import numpy as np


def local_indices(known_hours: np.ndarray, x: float, right: int) -> np.ndarray:
    """Start with the bracketing pair and add the nearest remaining points."""
    start, stop = right - 1, right + 1
    while stop - start < min(4, len(known_hours)):
        if stop == len(known_hours) or (
            start > 0 and x - known_hours[start - 1] <= known_hours[stop] - x
        ):
            start -= 1
        else:
            stop += 1
    return np.arange(start, stop)


def lagrange_interpolate(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("Lagrange interpolation needs at least two paired observations.")
    if len(set(xs)) != len(xs):
        raise ValueError("Lagrange interpolation requires distinct timestamps.")
    if not min(xs) <= x <= max(xs):
        raise ValueError("The target must be inside the known time range.")

    terms = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        basis = 1.0
        for j, xj in enumerate(xs):
            if i != j:
                basis *= (x - float(xj)) / (float(xi) - float(xj))
        terms.append(float(yi) * basis)
    return fsum(terms)
