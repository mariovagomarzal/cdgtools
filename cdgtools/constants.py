"""Module with the constants used in the package."""
from typing import Any, Union

import numpy as np
import sympy as sp

_AXIS_LABELS = {
    "x": 0,
    "y": 1,
    "z": 2,
}
"""Dictionary with the axis labels."""


_MATRIX_TYPES = Union[sp.ImmutableMatrix, sp.Matrix, np.ndarray, list[Any]]
"""Type hint for matrix-like objects."""
