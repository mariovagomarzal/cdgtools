"""Module containing utility functions."""
from typing import Any

import sympy as sp


def _are_equal_exprs(expr1: sp.Expr, expr2: sp.Expr, zero: Any = 0) -> bool:
    """Return True if expr1 and expr2 are equal, False otherwise."""
    return sp.simplify(expr1 - expr2) == zero


def _norm(vector: Any) -> sp.Expr:
    """Return the norm of a vector."""
    return sp.sqrt(vector.dot(vector))


def _is_bounded(set: sp.Set) -> bool:
    """Return True if set is bounded, False otherwise."""
    return set.inf is not sp.S.NegativeInfinity and \
        set.sup is not sp.S.Infinity


def _sols_in(
    eq_matrix: Any,
    symbol: sp.Symbol,
    domain: sp.Set
) -> sp.Set:
    """Return the solutions of `eq_matrix` in domain."""
    solutions = [sp.solveset(eq, symbol, domain=domain) for eq in eq_matrix]
    return sp.Intersection(*solutions)

def _has_sols_in(
    eq_matrix: Any,
    symbol: sp.Symbol,
    domain: sp.Set
) -> bool:
    """Return True if `eq_matrix` has solutions in domain, False otherwise."""
    return _sols_in(eq_matrix, symbol, domain).is_empty is False
