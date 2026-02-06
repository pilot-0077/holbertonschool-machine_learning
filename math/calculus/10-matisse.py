#!/usr/bin/env python3
"""function that derives a polynomial f(x)"""


def poly_derivative(poly):
    """Calculate the derivative of a polynomial"""

    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if len(poly) == 1:
        return [0]

    df = [i * poly[i] for i in range(1, len(poly))]

    if all(c == 0 for c in df):
        return [0]
    return df
