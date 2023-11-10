"""Module with the parametrizations classes."""
from __future__ import annotations
from typing import Any
import re

import sympy as sp

from cdgtools.constants import _AXIS_LABELS
from cdgtools._utils import (
    _norm,
    _are_equal_exprs,
    _is_bounded,
    _sols_in,
    _has_sols_in,
    _is_image_in_set,
    _is_invertible,
)


class Parametrization:
    """
    Base class for parametrizations.

    A parametrization is a continuous function that maps an interval of the
    real numbers to a set of points in a space. For example, the
    parametrization of a circle of radius $1$ in the plane is given by

    \\[
        \\gamma(t) = (\\cos(t), \\sin(t)),
    \\]

    where $t \\in [0, 2\\pi]$.

    Parameters
    ----------
    parametrization : Any
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
    parametrization : Any
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
    """
    parametrization: Any
    parameter: sp.Symbol
    domain: sp.Interval
    dimension: int

    def __init__(
        self,
        parametrization: Any,
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

    def __call__(self, t: Any) -> Any:
        """
        Evaluate the parametrization at the point `t`.

        Parameters
        ----------
        t : Any (but usually a number or an expression)
            The point at which to evaluate the parametrization.

        Returns
        -------
        point : Any
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
        if name in _AXIS_LABELS.keys():
            index = _AXIS_LABELS[name]
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
                are_equal = _are_equal_exprs(
                    self.parametrization,
                    other.parametrization.subs(other.parameter, self.parameter),
                    zero=sp.zeros(self.dimension, 1),
                )
                if are_equal and self.domain == other.domain:
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

    def diff(self, order: int = 1) -> Any:
        """
        Return the `order`-th derivative of the parametrization.

        Parameters
        ----------
        order : int
            Order of the derivative.

        Returns
        -------
        diff : Any
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

    def reparametrize(
        self,
        function: sp.Expr,
        new_parameter: sp.Symbol,
        new_domain: sp.Interval,
    ) -> Parametrization:
        """
        Reparametrize the parametrization.

        Reparametrize the parametrization by using the given function and
        domain. The function must be an invertible function from the new
        domain to the old domain.

        Parameters
        ----------
        function : Expr
            The function used to reparametrize the parametrization.
        new_domain : Interval
            The domain of the new parametrization.

        Returns
        -------
        reparametrization : Parametrization
            The reparametrization of the parametrization.

        Raises
        ------
        ValueError
            If the function is not invertible or the image of the new
            domain under the function is not a subset of the old domain.

        Examples
        --------
        We can reparametrize a parametrization by using the `reparametrize`
        method. The function must be an invertible function from the new
        domain to the old domain.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.reparametrize(t / 2, t, sp.Interval(0, 4 * sp.pi))
        Parametrization(Matrix([[cos(t/2)], [sin(t/2)]]), t, Interval(0, 4*pi))

        If the function is not invertible or the image of the new domain
        under the function is not a subset of the old domain, a
        `ValueError` is raised.

        >>> circle.reparametrize(t**2, t, sp.Interval(-1, 1))
        Traceback (most recent call last):
        ...
        ValueError: Function must be invertible.
        >>> circle.reparametrize(t, t, sp.Interval(0, 4 * sp.pi))
        Traceback (most recent call last):
        ...
        ValueError: Image of new domain must be a subset of the old domain.
        """
        if not _is_invertible(function, self.parameter, new_domain):
            raise ValueError(
                "Function must be invertible."
            )
        elif not _is_image_in_set(function, self.parameter, self.domain, new_domain):
            raise ValueError(
                "Image of new domain must be a subset of the old domain."
            )
        else:
            return Parametrization(
                parametrization=self.parametrization.subs(self.parameter, function),
                parameter=new_parameter,
                domain=new_domain,
            )

    def segment(self, new_interval: sp.Interval) -> Parametrization:
        """
        Return a segment of the parametrization.

        A segment of a parametrization is a parametrization defined in a
        closed and bounded interval of the domain of the parametrization.
        For example, if we have a parametrization of a circle in the plane
        defined in the interval $[0, 2\\pi]$, the parametrization

        \\[
            \\gamma(t) = (\\cos(t), \\sin(t)),
        \\]

        with $t \\in [0, \\pi]$, is a segment of the parametrization of the
        circle.

        Parameters
        ----------
        new_interval : Interval
            The interval in which to define the segment.

        Returns
        -------
        segment : Parametrization
            The segment of the parametrization.

        Raises
        ------
        ValueError
            If `new_interval` is not a closed and bounded interval.

        Examples
        --------
        We can get a segment of a parametrization by using the `segment`
        method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.segment(sp.Interval(0, sp.pi))
        Parametrization(Matrix([[cos(t)], [sin(t)]]), t, Interval(0, pi))

        If the interval is not closed and bounded, a `ValueError` is
        raised.

        >>> circle.segment(sp.Interval(0, sp.oo))
        Traceback (most recent call last):
        ...
        ValueError: Interval must be closed and bounded, not Interval(0, oo).
        """
        if not (_is_bounded(new_interval) and new_interval.is_closed):
            raise ValueError(
                f"Interval must be closed and bounded, not {new_interval}."
            )

        return Parametrization(
            parametrization=self.parametrization,
            parameter=self.parameter,
            domain=new_interval,
        )

    def is_closed(self) -> bool:
        """
        Check if the curve is closed.

        A curve is closed if its parametrization is defined in a closed and
        bounded interval and the value of the parametrization at the
        extremes of the interval is the same, i.e., if

        \\[
            \\gamma(a) = \\gamma(b),
        \\]

        with $\\gamma: [a, b] \\longrightarrow \\mathbb{R}^n$.

        Returns
        -------
        closed : bool
            Whether the curve is closed.

        Examples
        --------
        We can check if a curve is closed by using the `is_closed` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.is_closed()
        True

        If the parametrization is not defined in a closed and bounded
        interval, the curve is not closed.

        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([t, t**2]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> other.is_closed()
        False

        If the parametrization is defined in a closed and bounded interval
        but the value of the parametrization at the extremes of the
        interval is not the same, the curve is not closed.

        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, sp.pi),
        ... )
        >>> other.is_closed()
        False
        """
        if not (_is_bounded(self.domain) and self.domain.is_closed):
            return False
        else:
            return _are_equal_exprs(
                self(self.domain.inf),
                self(self.domain.sup),
                zero=sp.zeros(self.dimension, 1),
            )

    def velocity(self) -> Any:
        """
        Return the velocity vector of the parametrization.

        The velocity of a parametrization is the derivative of the
        parametrization with respect to the parameter, i.e.,

        \\[
            \\gamma'(t).
        \\]

        Returns
        -------
        velocity : Any
            The velocity of the parametrization.

        Examples
        --------
        We can get the velocity of a parametrization by using the
        `velocity` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.velocity()
        Matrix([
        [-sin(t)],
        [ cos(t)]])
        """
        return self.diff()

    def speed(self) -> sp.Expr:
        """
        Return the speed of the parametrization.

        The speed of a parametrization is the norm of its velocity, i.e.,

        \\[
            \\|\\gamma'(t)\\|.
        \\]

        Returns
        -------
        speed : Expr
            The speed of the parametrization.

        Examples
        --------
        We can get the speed of a parametrization by using the `speed`
        method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.speed()
        sqrt(sin(t)**2 + cos(t)**2)
        >>> circle.speed().simplify()
        1
        """
        return _norm(self.velocity())

    def tangent(self) -> Any:
        """
        Return the tangent vector of the parametrization.

        The tangent vector of a parametrization is the unit vector in the
        direction of the velocity of the parametrization, i.e.,

        \\[
            \\vec{t} = \\frac{\\gamma'(t)}{\\|\\gamma'(t)\\|}.
        \\]

        Returns
        -------
        tangent : Any
            The tangent vector of the parametrization.

        Examples
        --------
        We can get the tangent vector of a parametrization by using the
        `tangent` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> line = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([t, t]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> line.tangent()
        Matrix([
        [sqrt(2)/2],
        [sqrt(2)/2]])
        """
        return self.velocity() / self.speed()

    def is_regular(self, subs: dict[sp.Symbol, Any] = {}) -> bool:
        """
        Check if the parametrization is regular.

        A parametrization is regular if its derivative is not the zero
        vector in any point of its domain or, equivalently, if the speed
        of the parametrization is not zero in any point of its domain.

        To check if the parametrization is regular, we solve the equation

        \\[
            \\|\\gamma'(t)\\| = 0.
        \\]

        over the domain of the parametrization. If the equation has no
        solutions, the parametrization is regular. If the equation has
        solutions, the parametrization is not regular.

        Since the expressions of the parametrization and its derivative
        may contain other symbols, we have to specify numerical values for
        those symbols. We do that by using the `subs` argument.

        In some cases, we can't solve the equation analytically. In that
        case, a `NotImplementedError` is raised.

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
        speed = self.speed().subs(subs)
        free_symbols = speed.free_symbols - {self.parameter}
        if free_symbols != set():
            raise ValueError("Substitutions must be specified.")

        return not _has_sols_in([speed], self.parameter, self.domain)

    def is_natural(self, subs: dict[sp.Symbol, Any] = {}) -> bool:
        """
        Check if the parametrization is natural.

        A parametrization is natural if its speed is $1$ in any point of
        its domain. To check if the parametrization is natural, we solve
        the equation

        \\[
            \\|\\gamma'(t)\\| = 1.
        \\]

        over the domain of the parametrization. If the whole domain is
        solution of the equation, the parametrization is natural.
        Otherwise, it is not natural.

        Since the expressions of the parametrization and its derivative
        may contain other symbols, we have to specify numerical values for
        those symbols. We do that by using the `subs` argument.

        In some cases, we can't solve the equation analytically. In that
        case, a `NotImplementedError` is raised.

        Parameters
        ----------
        subs : dict[Symbol, Any] or None
            Substitutions to make before checking if the parametrization is
            natural.

        Returns
        -------
        natural : bool
            Whether the parametrization is natural.

        Raises
        ------
        ValueError
            If the first derivative of the parametrization contains other
            symbols and no substitutions are specified.
        NotImplementedError
            If the equation :math:`\\gamma'(t) = 1` can't be solved
            analytically.

        Examples
        --------
        We can check if a parametrization is natural by using the
        `is_natural` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.is_natural()
        True

        If our parametrization contains other symbols, we have to specify
        numerical values for those symbols. We can check that depending on
        the value of the parameter, the parametrization is natural or not.

        >>> lamda = sp.symbols("lambda")
        >>> other = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([lamda * t, lamda * t]),
        ...     parameter=t,
        ...     domain=sp.Reals,
        ... )
        >>> other.is_natural({lamda: 1})
        False
        >>> other.is_natural({lamda: 1/sp.sqrt(2)})
        True

        If no substitutions are specified and the first derivative of the
        parametrization contains other symbols, a `ValueError` is raised.

        >>> other.is_natural()
        Traceback (most recent call last):
        ...
        ValueError: Substitutions must be specified.
        """
        speed = self.speed().subs(subs)
        free_symbols = speed.free_symbols - {self.parameter}
        if free_symbols != set():
            raise ValueError("Substitutions must be specified.")

        return _sols_in([speed - 1], self.parameter, self.domain) == self.domain

    def curve_length(self, interval: sp.Interval | None = None) -> sp.Expr:
        """
        Computes the length of the parametrization in `interval`.
        Note that if we reparametrize the curve, the curve length does not change.

        Parameters
        ----------
        interval : sp.Interval
            The interval where we compute the arc length.

        Returns
        -------
        lenght : float
            The length of the curve.

        Raises
        ------
        ValueError
            If `interval` is not contained in the Parametrization's domain.

        Examples
        --------
        We can get the arc length of a parametrization by using the
        `curve_length` method.

        >>> from cdgtools import Parametrization
        >>> import sympy as sp
        >>> t = sp.symbols("t")
        >>> circle = Parametrization(
        ...     parametrization=sp.ImmutableMatrix([sp.cos(t), sp.sin(t)]),
        ...     parameter=t,
        ...     domain=sp.Interval(0, 2 * sp.pi),
        ... )
        >>> circle.curve_length()
        2*pi

        If we specify the interval, we can get the arc length of the
        parametrization in that interval.

        >>> circle.curve_length(sp.Interval(0, sp.pi))
        pi

        If the interval is not contained in the parametrization's domain,
        a `ValueError` is raised.

        >>> circle.curve_length(sp.Interval(0, 3 * sp.pi))
        Traceback (most recent call last):
        ...
        ValueError: The specified interval must be a subset of the parametrization's domain.
        """
        if interval is None:
            interval = self.domain
        elif not interval.is_subset(self.domain):
            raise ValueError(
                "The specified interval must be a subset of the parametrization's domain."
            )

        a, b = interval.inf, interval.sup
        return sp.integrate(self.speed(), (self.parameter, a, b))


class Parametrization2D(Parametrization):
    pass


class Parametrization3D(Parametrization):
    pass
