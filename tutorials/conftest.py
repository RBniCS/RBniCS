# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import re
import sys
import importlib
import pytest
import pytest_flake8
from nbconvert.exporters import PythonExporter
import nbconvert.filters
from mpi4py import MPI
try:
    import dolfin  # otherwise the next import from rbnics would disable dolfin as a required backend  # noqa: F401
except ImportError:
    pass
from rbnics.utils.test import (
    add_gold_options, disable_matplotlib, enable_matplotlib, process_gold_options, run_and_compare_to_gold)


def pytest_addoption(parser):
    add_gold_options(parser)


def pytest_configure(config):
    process_gold_options(config)


def pytest_ignore_collect(path, config):
    if path.ext == ".py" and path.new(ext=".ipynb").exists():  # ignore .py files obtained from previous runs
        return True
    else:
        return False


def pytest_collect_file(path, parent):
    """
    Collect tutorial files.
    """
    if path.ext == ".ipynb":
        # Convert .ipynb notebooks to plain .py files
        def comment_lines(text, prefix="# "):
            regex = re.compile(r".{1,80}(?:\s+|$)")
            input_lines = text.split("\n")
            output_lines = [split_line.rstrip() for line in input_lines for split_line in regex.findall(line)]
            output = prefix + ("\n" + prefix).join(output_lines)
            return output.replace(prefix + "\n", prefix.rstrip(" ") + "\n")

        def ipython2python(code):
            return nbconvert.filters.ipython2python(code).rstrip("\n") + "\n"

        filters = {
            "comment_lines": comment_lines,
            "ipython2python": ipython2python
        }
        exporter = PythonExporter(filters=filters)
        exporter.exclude_input_prompt = True
        code, _ = exporter.from_filename(path)
        code = code.rstrip("\n") + "\n"
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(path.new(ext=".py"), "w", encoding="utf-8") as f:
                f.write(code)
        MPI.COMM_WORLD.Barrier()
        # Collect the corresponding .py file
        config = parent.config
        if config.getoption("--flake8"):
            return pytest_flake8.pytest_collect_file(path.new(ext=".py"), parent)
        else:
            if "data" not in path.dirname:  # skip running mesh generation notebooks
                if not path.basename.startswith("x"):
                    return TutorialFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
                else:
                    return DoNothingFile.from_parent(parent=parent, fspath=path.new(ext=".py"))
    elif path.ext == ".py":  # TODO remove after transition to ipynb is complete? assert never py files?
        if (path.basename not in "conftest.py"  # do not run pytest configuration file
                or "data" not in path.dirname):  # skip running mesh generation notebooks
            if not path.basename.startswith("x"):
                return TutorialFile.from_parent(parent=parent, fspath=path)
            else:
                return DoNothingFile.from_parent(parent=parent, fspath=path)


def pytest_pycollect_makemodule(path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """
    if path.ext == ".py":
        assert not path.new(ext=".ipynb").exists(), "Please run pytest on jupyter notebooks, not plain python files."
        return DoNothingFile.from_parent(
            parent=parent, fspath=path)  # TODO remove after transition to ipynb is complete?


def pytest_runtest_teardown(item, nextitem):
    # Do the normal teardown
    item.teardown()
    # Add a MPI barrier in parallel
    MPI.COMM_WORLD.Barrier()


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem.from_parent(parent=self, name=os.path.relpath(str(self.fspath), str(self.parent.fspath)))


class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial.
    """

    @run_and_compare_to_gold()
    def runtest(self):
        disable_matplotlib()
        os.chdir(self.parent.fspath.dirname)
        sys.path.append(self.parent.fspath.dirname)
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.fspath))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        enable_matplotlib()

    def reportinfo(self):
        return self.fspath, 0, self.name


class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice python files explicitly provided on the command line.
    """

    def collect(self):
        return []
