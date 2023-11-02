"""Module with the parametrizations classes."""
from typing import Any, Union
import re

import numpy as np
import sympy as sp

from cdgtools.constants import AXIS_LABELS


# Constants
_MATRIX_TYPES = Union[sp.ImmutableMatrix, sp.Matrix, np.ndarray, list[Any]]


class Parametrization:
    """
    Base class for parametrizations.

    Explanation
    -----------
    A parametrization is a function that maps an interval of the real numbers
    to a set of points in a space. For example, the parametrization of a
    circle of radius 1 in the plane is given by:

    .. math::

        \\gamma(t) = (\\cos(t), \\sin(t)),

    where :math:`t \\in [0, 2\\pi]`.

    Parameters
    ----------
    parametrization : ImmutableMatrix (or any other matrix type)
        The parametrization of the space.
    parameter : Symbol
        The symbol used to represent the parameter.
    domain : Interval
        The domain of the parameter.

    Examples
    --------
    >>> from cdgtools import Parametrization
    >>> import sympy as sp
    >>> t = sp.symbols("t")
    >>> circle = Parametrization(
    ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
    ...     parameter=t,
    ...     domain=sp.Interval(0, 2 * sp.pi),
    ... )
    >>> circle.parametrization
    Matrix([
    [cos(t)],
    [sin(t)]])
    >>> circle.parameter
    t
    >>> circle.domain
    Interval(0, 2*pi)
    >>> circle.dimension
    2
    """
    parametrization: sp.ImmutableMatrix
    parameter: sp.Symbol
    domain: sp.Interval
    dimension: int

    def __init__(
        self,
        parametrization: _MATRIX_TYPES,
        parameter: sp.Symbol,
        domain: sp.Interval = sp.Reals,
    ) -> None:
        self.parametrization = sp.ImmutableMatrix(parametrization)
        self.parameter = parameter
        self.domain = domain
        self.dimension = self.parametrization.shape[0]

    def __str__(self) -> str:
        return f"{self.parametrization}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"\
               f"({self.parametrization}, {self.parameter}, {self.domain})"

    def __call__(self, t: Any) -> sp.ImmutableMatrix:
        """
        Evaluate the parametrization at the point `t`.

        Parameters
        ----------
        t : Any (but usually a number or an expression)
            The point at which to evaluate the parametrization.

        Returns
        -------
        point : ImmutableMatrix
            The point in the space corresponding to t.

        Raises
        ------
        ValueError
            If `t` is not in the domain of the parametrization.

        Examples
        --------
        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=[0, 2 * sp.pi],
        ... )
        >>> circle(0)
        Matrix([
        [1],
        [0]])
        >>> circle(10) # not in the domain
        Traceback (most recent call last):
        ...
        ValueError: t must be in the interval [0, 2*pi], not 10.
        """
        try:
            if t not in self.domain:
                raise ValueError(
                    f"t must be in the interval {self.domain}, not {t}."
                )
        except TypeError:
            pass

        return self.parametrization.subs(self.parameter, t)

    def __getattr__(self, name: str) -> sp.Expr:
        """
        Get the `name`-th coordinate of the parametrization.

        Explanation
        -----------
        This method is used to get the `name`-th coordinate of the
        parametrization. For example, if we have a parametrization of a
        circle in the plane, we can get the `x` and `y` coordinates of the
        parametrization by doing `circle.x` and `circle.y`, respectively.
        Also, if the parametrization is in a space of dimension greater
        than 3, we can get the `x1`, `x2`, `x3`, etc. coordinates, where
        `x` could be any letter.

        Parameters
        ----------
        name : str
            The name of the coordinate to get.

        Returns
        -------
        coordinate : Expr
            The `name`-th coordinate of the parametrization.

        Raises
        ------
        AttributeError
            If `name` is not a valid coordinate.
        ValueError
            If `name` is a valid coordinate but the parametrization is not
            in a space of dimension greater than or equal to the index of
            the coordinate.

        Examples
        --------
        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=[0, 2 * sp.pi],
        ... )
        >>> circle.x
        cos(t)
        >>> circle.y
        sin(t)
        >>> circle.z
        Traceback (most recent call last):
        ...
        ValueError: Index must be less than 2, not 2.
        """
        if name in AXIS_LABELS.keys():
            index = AXIS_LABELS[name]
        else:
            # Chek if `name` matches some string of the form `[a-z]\d+`
            # (e.g. `x1`, `y2`, `z3`, etc.)
            match = re.match(r"[a-z]\d+", name)
            if match:
                index = int(match.group()[1:])
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'."
                )

        if index < self.dimension:
            return self.parametrization[index]
        else:
            raise ValueError(
                f"Index must be less than {self.dimension}, not {index}."
            )


class Parametrization2D(Parametrization):
    pass


class Parametrization3D(Parametrization):
    pass
