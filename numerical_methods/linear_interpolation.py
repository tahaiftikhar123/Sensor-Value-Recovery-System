"""Interpolate between two original observations."""


def linear_interpolate(x: float, x0: float, y0: float, x1: float, y1: float) -> float:
    if x1 <= x0:
        raise ValueError("Linear interpolation requires two distinct, ordered timestamps.")
    if not x0 <= x <= x1:
        raise ValueError("The target must be inside the known time range.")
    fraction = (x - x0) / (x1 - x0)
    # The convex form avoids overflow in y1 - y0 for extreme opposite readings.
    return (1.0 - fraction) * y0 + fraction * y1
