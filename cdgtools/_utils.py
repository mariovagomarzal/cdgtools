"""Module containing utility functions."""
from typing import Any

import sympy as sp


def _has_sols_in(
    eq_matrix: Any,
    symbol: sp.Symbol,
    domain: sp.Set
) -> bool:
    """Return True if `eq_matrix` has solutions in domain, False otherwise."""
    solutions = [sp.solveset(eq, symbol, domain=domain) for eq in eq_matrix]
    return not sp.Intersection(*solutions).is_empty

def _is_invertible(f: sp.Expr, symbol: sp.Symbol, domain: sp.Set) -> bool:
    """
    Return True if `f` is invertible, False otherwise.
       
    We use the mathematical fact that a function is invertible if and only 
    if its derivative is not zero in all the point of the domain.
    """
    return not _has_sols_in([sp.diff(f,symbol)], symbol, domain)

def _is_image_in_set(
        f: sp.Expr,
        symbol: sp.Symbol,
        old_interval: sp.Interval,
        new_interval: sp.Interval
) -> bool:
    """Return True if the image of `new_interval` by `f` is contained in `old_interval`, False otherwise."""
    return sp.imageset(symbol, f, new_interval).is_subset(old_interval)