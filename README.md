# cdgtools

A Python module for studying curves, surfaces and other concepts from
Clasic Differential Geometry.

## Contributing

### Development Environment

The development environment is managed with [Poetry][poetry]. To install
the dependencies, run:

```bash
poetry install
```

Then, to activate the virtual environment, run:

```bash
poetry shell
```

We are using [pre-commit][pre-commit] to manage the git hooks. To
install them, run:

```bash
pre-commit install
```

### Testing and Linting

We are using [Invoke][invoke] to manage the tasks. Testing is done with
[pytest][pytest] and the [pytest-doctestplus][doctestplus] plugin. The
linter is [Ruff][ruff] and [Mypy][mypy] for type checking.

Once the development environment is set up, you can run the tests with:

```bash
inv tests
```

You can add the optional flag `--no-doctest` to skip the doctests. Also,
you can add `-v` with up to three `v`s to increase the verbosity of the
output.

To run the linter, run:

```bash
inv lint
```

You can add the optional flag `--fix` to automatically fix the linting
errors that can be fixed automatically.

### Docstrings guidelines

We are using the [Numpy docstring guidelines][numpy-docstring] for the
docstrings.

## License

This software is licensed by [Pedro Pasalados Guiral][pedro] and [Mario
Vago Marzal][mario] under the terms of the [MIT License](/LICENSE).


[poetry]: https://python-poetry.org/
[pre-commit]: https://pre-commit.com/
[invoke]: https://www.pyinvoke.org/
[pytest]: https://docs.pytest.org/en/stable/
[doctestplus]: https://github.com/scientific-python/pytest-doctestplus
[ruff]: https://docs.astral.sh/ruff/
[mypy]: https://mypy.readthedocs.io/en/stable/
[numpy-docstring]: https://numpydoc.readthedocs.io/en/latest/format.html
[pedro]: https://github.com/pedropasa03
[mario]: https://github.com/mariovagomarzal
