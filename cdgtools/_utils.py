"""Module containing utility functions."""
from typing import Any

import sympy as sp


# Basic functions to manipulate expressions
def _are_equal_exprs(expr1: sp.Expr, expr2: sp.Expr, zero: Any = 0) -> bool:
    """Return True if expr1 and expr2 are equal, False otherwise."""
    return sp.simplify(expr1 - expr2) == zero

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


# Algebraic functions
def _norm(vector: Any) -> sp.Expr:
    """Return the norm of a vector."""
    return sp.sqrt(vector.dot(vector))


# Sets and mappings functions
def _is_bounded(set: sp.Set) -> bool:
    """Return True if set is bounded, False otherwise."""
    return set.inf is not sp.S.NegativeInfinity and \
        set.sup is not sp.S.Infinity

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
    """
    Return True if the image of `new_interval` by `f` is contained in
    `old_interval`, False otherwise.
    """
    return sp.imageset(symbol, f, new_interval).is_subset(old_interval)
