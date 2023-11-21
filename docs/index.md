# cdgtools

A Python module for studying curves, surfaces and other concepts from
Classical Differential Geometry.

This module started as an small project aimed to perform some basic
symbolic computations for the subject of Classical Differential Geometry
imparted at the University of Valencia. However, it has grown to be a more
general purpose tool for studying curves and surfaces with several
graphical and symbolic capabilities.

To see quick examples of what can be done with `cdgtools`, check the
[examples](examples.md) page. For a more detailed guide on how to install
and use `cdgtools`, check the [usage](usage/index.md) page.

## For symbolic computation

The `cdgtools` module provides a set of classes and functions for
representing curves and surfaces parametrizations. With these objects,
it is possible to perform symbolic computations to study properties of
curves and surfaces, such as their curvature, torsion, etc.

## For LaTeX rendering

Since LaTeX is the standard for writing mathematical documents, the
`cdgtools` module provides a set of functions for rendering its objects
as LaTeX math expressions. Moreover, we also provide a set of functions
to plot curves and surfaces using LaTeX's `tikz` package.

## For animations

One of the best ways to visualize curves and surfaces, apart from plotting
them, is to animate them. Lukily, in Python we have available the
[manim](https://www.manim.community) library, which allows us to create
beautiful animations of mathematical objects.

The `cdgtools` module provides a set of utilities to create animations
of curves and surfaces using `manim`.
