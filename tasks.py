"""Invoke tasks."""
from invoke import task
import os


# Constants
PROJECT_NAME = "cdgtools"
PTY_MODE = True


# Disable `PTY_MODE` if using Windows
if os.name == 'nt':
    PTY_MODE = False


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

    c.run(pytest_command, pty=PTY_MODE)


@task
def lint(c, fix=False):
    """Lint code."""
    if fix:
        ruff_command = "ruff check --fix ."
    else:
        ruff_command = "ruff check ."

    print("Ruff linting...")
    c.run(ruff_command, pty=PTY_MODE, warn=True)
    print("Mypy linting...")
    c.run("mypy cdgtools", pty=PTY_MODE)


@task
def serve(c):
    """Serve documentation."""
    c.run("mkdocs serve", pty=PTY_MODE)


@task
def build(c):
    """Build documentation."""
    c.run("mkdocs build --clean", pty=PTY_MODE)
