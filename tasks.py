"""Invoke tasks."""
from invoke import task


# Constants
PROJECT_NAME = "cdgtools"


@task(incrementable=["verbose"])
def tests(c, verbose=0, no_doctest=False):
    """Run tests."""
    verbose_level = "v" * verbose
    if verbose_level:
        pytest_command = f"pytest -{verbose_level}"
    else:
        pytest_command = "pytest"

    if not no_doctest:
        pytest_command += " --doctest-modules"

    c.run(pytest_command, pty=True)


@task
def lint(c, fix=False):
    """Lint code."""
    if fix:
        ruff_command = "ruff check --fix ."
    else:
        ruff_command = "ruff check ."

    print("Ruff linting...")
    c.run(ruff_command, pty=True, warn=True)
    print("Mypy linting...")
    c.run("mypy cdgtools", pty=True)
