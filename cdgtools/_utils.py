"""Module containing utility functions."""
from typing import Any

import sympy as sp


def _has_sols_in(
    eq_matrix: Any,
    symbol: sp.Symbol,
    domain: sp.Set
) -> bool:
    """Return True if f has solutions in domain, False otherwise."""
    solutions = [sp.solveset(eq, symbol, domain=domain) for eq in eq_matrix]
    return sp.Intersection(*solutions).is_empty is False
