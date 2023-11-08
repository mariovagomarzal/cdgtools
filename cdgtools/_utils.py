"""Module containing utility functions."""
import sympy as sp


def _has_sols_in(f: sp.Expr, x: sp.Symbol, domain: sp.Set) -> bool:
    """Return True if f has solutions in domain, False otherwise."""
    return sp.solveset(f, x, domain=domain).is_empty is False
