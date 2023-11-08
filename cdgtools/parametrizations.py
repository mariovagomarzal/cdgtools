"""Module with the parametrizations classes."""
from __future__ import annotations
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

    Raises
    ------
    ValueError
        If `parametrization` is not a column vector.

    Attributes
    ----------
    parametrization : ImmutableMatrix
        The parametrization of the space.
    parameter : Symbol
        The symbol used to represent the parameter.
    domain : Interval
        The domain of the parameter.
    dimension : int
        The dimension of the space.

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

    See Also
    --------
    Parametrization2D : Class for 2D parametrizations.
    Parametrization3D : Class for 3D parametrizations.
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
        self.dimension, cols = self.parametrization.shape
        if cols != 1:
            raise ValueError(
                f"Parametrization must be a column vector, not {cols} columns."
            )

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
        We can evaluate the parametrization at a point by calling the
        parametrization object with the point as an argument.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle(0)
        Matrix([
        [1],
        [0]])

        If the point is not in the domain, a `ValueError` is raised.

        >>> circle(10) # not in the domain
        Traceback (most recent call last):
        ...
        ValueError: t must be in Interval(0, 2*pi), not 10.

        We can also evaluate the parametrization at an expression. Domain
        checking is ignored in this case.

        >>> lamda = sp.symbols("lambda")
        >>> circle(2*lamda)
        Matrix([
        [cos(2*lambda)],
        [sin(2*lambda)]])
        """
        try:
            if t not in self.domain:
                raise ValueError(
                    f"t must be in {self.domain}, not {t}."
                )
        except TypeError:
            pass

        return self.parametrization.subs(self.parameter, t)

    def __getitem__(self, index: int) -> sp.Expr:
        """
        Get the `index`-th coordinate of the parametrization.

        This method is used to get the `index`-th coordinate of the
        parametrization. For example, if we have a parametrization of a
        circle in the plane, we can get the `x` and `y` coordinates of the
        parametrization by doing `circle[0]` and `circle[1]`, respectively.

        Parameters
        ----------
        index : int
            The index of the coordinate to get.

        Returns
        -------
        coordinate : Expr
            The `index`-th coordinate of the parametrization.

        Raises
        ------
        ValueError
            If `index` is not a valid coordinate.

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
        >>> circle[0]
        cos(t)
        >>> circle[1]
        sin(t)
        >>> circle[2]
        Traceback (most recent call last):
        ...
        ValueError: Index must be less than 2, not 2.
        """
        if index < self.dimension:
            return self.parametrization[index]
        else:
            raise ValueError(
                f"Index must be less than {self.dimension}, not {index}."
            )

    def __getattr__(self, name: str) -> sp.Expr:
        """
        Get the `name`-th coordinate of the parametrization.

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
        We can access the coordinates of the parametrization by using the
        `x`, `y`, or `z` attributes.

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

        We can also access the coordinates of parametrizations using the
        `x1`, `x2`, `x3`, etc. attributes. This is useful when the
        parametrization is in a space of dimension greater than 3. Note
        that the letter `x` can be any letter.

        >>> circle.x1
        cos(t)
        >>> circle.y2
        sin(t)
        >>> circle.z3
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
                index = int(match.group()[1:]) - 1
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'."
                )

        return self[index]

    def __eq__(self, other: Any) -> bool:
        """
        Check if two parametrizations are equal.

        For to parametrizations to be equal, they must have the same
        parametrization and domain. The parameter can be different.
        To check that, we substitute the parameter of the other
        parametrization by the parameter of this parametrization and
        check if the resulting parametrizations are equal by checking that
        its difference is the zero vector.

        Parameters
        ----------
        other : Any
            The object to compare to.

        Returns
        -------
        equal : bool
            Whether the two parametrizations are equal.

        Examples
        --------
        We can check if two parametrizations are equal by using the `==`
        operator. It doesn't matter if the parameter is different or if the
        parametrization forms are different (e.g. expanded or binomial).

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> s = sp.symbols("s")
        >>> parabola_binom = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([t, (t + 1)**2]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> parabola_expanded = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([s, s**2 + 2*s + 1]),
        ...     parameter=s,
        ...     domain=sp.Reals,
        ... )
        >>> parabola_binom == parabola_expanded
        True

        If the parametrizations have different dimensions, expressions or
        domains, they are not equal.

        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=[0, 2 * sp.pi],
        ... )
        >>> circle == parabola_binom
        False

        A parametrization can only be equal to another parametrization.

        >>> circle == 1
        False
        """
        if not isinstance(other, Parametrization):
            return False
        else:
            if self.dimension != other.dimension:
                return False
            else:
                difference = self.parametrization - other.parametrization.subs(
                    other.parameter, self.parameter
                )
                if difference.expand() == sp.zeros(self.dimension, 1) and \
                    self.domain == other.domain:
                    return True
                else:
                    return False

    def __add__(self, other: Parametrization) -> Parametrization:
        """
        Add two parametrizations.

        Add two parametrizations by adding their parametrization vectors.
        Dimensions must be equal. If the parameters are different, the
        parameter of the first parametrization is used. If the domains are
        different, the most restrictive domain, i. e., the intersection of
        both domains, is used.

        Parameters
        ----------
        other : Parametrization
            The parametrization to add.

        Returns
        -------
        parametrization : Parametrization
            The sum of the two parametrizations.

        Raises
        ------
        ValueError
            If the dimensions of the parametrizations are different.

        Examples
        --------
        We can add two parametrizations with the same dimension, parameter
        and domain by using the `+` operator.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle + circle
        Parametrization(Matrix([[2*cos(t)], [2*sin(t)]]), t, Interval(0, 2*pi))

        If the dimensions of the parametrizations are different, a
        `ValueError` is raised.

        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([t, t**2, t**3]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> circle + other
        Traceback (most recent call last):
        ...
        ValueError: Dimension of both parametrizations must be equal, not 2 and 3.

        If the parameters of the parametrizations are different, the
        parameter of the first parametrization is used.

        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=sp.symbols("s"),
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> sum_param = circle + other
        >>> sum_param.parameter
        t

        If the domains of the parametrizations are different, the most
        restrictive domain is used.

        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, sp.pi),
        ... )
        >>> sum_param = circle + other
        >>> sum_param.domain
        Interval(0, pi)
        """

        if self.dimension != other.dimension:
            raise ValueError(
                f"Dimension of both parametrizations must be equal, not"
                f" {self.dimension} and {other.dimension}."
            )
        else:
            domain = self.domain
            if self.domain != other.domain:
                domain = self.domain.intersect(other.domain)

            return Parametrization(
                parametrization=self.parametrization + other.parametrization,
                parameter=self.parameter,
                domain=domain,
            )

    def __neg__(self) -> Parametrization:
        """Additive inverse of the parametrization."""
        return Parametrization(
            parametrization=-self.parametrization,
            parameter=self.parameter,
            domain=self.domain,
        )

    def __sub__(self, other: Parametrization) -> Parametrization:
        """Subtract two parametrizations."""
        return self + (-other)

    def diff(self, order: int = 1) -> sp.ImmutableMatrix:
        """
        Return the `order`-th derivative of the parametrization.

        Parameters
        ----------
        order : int
            Order of the derivative.

        Returns
        -------
        diff : ImmutableMatrix
            The `order`-th derivative of the parametrization.

        Examples
        --------
        We can get the first derivative of a parametrization by using the
        `diff` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.diff()
        Matrix([
        [-sin(t)],
        [ cos(t)]])

        If we specify the order of the derivative, we can get the second
        derivative.

        >>> circle.diff(2)
        Matrix([
        [-cos(t)],
        [-sin(t)]])
        """
        return self.parametrization.diff(self.parameter, order)

    def is_regular(self, subs: dict[sp.Symbol, Any] = {}) -> bool:
        """
        Check if the parametrization is regular.

        A parametrization is regular if its derivative is not the zero
        vector in any point of its domain. For that, we calculate the
        derivative of the parametrization and check that the equation
        :math:`\\gamma'(t) = 0` has no solution in the domain of the
        parametrization.

        Since the expressions of the parametrization and its derivative
        may contain other symbols, we have to specify numerical values for
        those symbols. We do that by using the `subs` argument.

        In some cases, we can't solve the equation :math:`\\gamma'(t) = 0`
        analytically. In that case, a `NotImplementedError` is raised.

        Parameters
        ----------
        subs : dict[Symbol, Any] or None
            Substitutions to make before checking if the parametrization is
            regular.

        Returns
        -------
        regular : bool
            Whether the parametrization is regular.

        Raises
        ------
        ValueError
            If the first derivative of the parametrization contains other
            symbols and no substitutions are specified.
        NotImplementedError
            If the equation :math:`\\gamma'(t) = 0` can't be solved
            analytically.

        Examples
        --------
        We can check if a parametrization is regular by using the
        `is_regular` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.is_regular()
        True

        If our parametrization contains other symbols, we have to specify
        numerical values for those symbols. We can check that depending on
        the value of the parameter, the parametrization is regular or not.

        >>> lamda = sp.symbols("lambda")
        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([t**2, lamda * sp.exp(t)]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> other.is_regular({lamda: 1})
        True
        >>> other.is_regular({lamda: 0})
        False

        If no substitutions are specified and the first derivative of the
        parametrization contains other symbols, a `ValueError` is raised.

        >>> other.is_regular()
        Traceback (most recent call last):
        ...
        ValueError: Substitutions must be specified.
        """
        derivative = self.diff().subs(subs)
        free_symbols = derivative.free_symbols - {self.parameter}
        if free_symbols != set():
            raise ValueError("Substitutions must be specified.")

        solutions = [sp.solveset(component, self.parameter) for component in derivative]
        solution = sp.Intersection(self.domain, *solutions)

        return solution == sp.EmptySet


class Parametrization2D(Parametrization):
    pass


class Parametrization3D(Parametrization):
    pass
