# Copyright (C) 2015-2022 by the RBniCS authors
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


def pytest_ignore_collect(collection_path, path, config):
    if collection_path.suffix == ".py" and collection_path.with_suffix(".ipynb").exists():
        # ignore .py files obtained from previous runs
        return True
    else:
        return False


def pytest_collect_file(file_path, path, parent):
    """
    Collect tutorial files.
    """
    if file_path.suffix == ".ipynb":
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
        code, _ = exporter.from_filename(file_path)
        code = code.rstrip("\n") + "\n"
        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(file_path.with_suffix(".py"), "w", encoding="utf-8") as f:
                f.write(code)
        MPI.COMM_WORLD.Barrier()
        # Collect the corresponding .py file
        config = parent.config
        if config.getoption("--flake8"):
            return pytest_flake8.pytest_collect_file(file_path.with_suffix(".py"), None, parent)
        else:
            if "data" not in str(file_path.parent):  # skip running mesh generation notebooks
                if not file_path.name.startswith("x"):
                    return TutorialFile.from_parent(parent=parent, path=file_path.with_suffix(".py"))
                else:
                    return DoNothingFile.from_parent(parent=parent, path=file_path.with_suffix(".py"))
    elif file_path.suffix == ".py":  # TODO remove after transition to ipynb is complete? assert never py files?
        if (file_path.name not in "conftest.py"  # do not run pytest configuration file
                or "data" not in str(file_path.parent)):  # skip running mesh generation notebooks
            if not file_path.name.startswith("x"):
                return TutorialFile.from_parent(parent=parent, path=file_path)
            else:
                return DoNothingFile.from_parent(parent=parent, path=file_path)


def pytest_pycollect_makemodule(module_path, path, parent):
    """
    Disable running .py files produced by previous runs, as they may get out of sync with the corresponding .ipynb file.
    """
    if module_path.suffix == ".py":
        assert not module_path.with_suffix(".ipynb").exists(), (
            "Please run pytest on jupyter notebooks, not plain python files.")
        return DoNothingFile.from_parent(
            parent=parent, path=module_path)  # TODO remove after transition to ipynb is complete?


def pytest_runtest_teardown(item, nextitem):
    # Add a MPI barrier in parallel
    MPI.COMM_WORLD.Barrier()


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files.
    """

    def collect(self):
        yield TutorialItem.from_parent(parent=self, name=os.path.relpath(str(self.path), str(self.parent.path)))


class TutorialItem(pytest.Item):
    """
    Handle the execution of the tutorial.
    """

    @run_and_compare_to_gold()
    def runtest(self):
        disable_matplotlib()
        os.chdir(self.parent.path.parent)
        sys.path.append(str(self.parent.path.parent))
        spec = importlib.util.spec_from_file_location(self.name, str(self.parent.path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        enable_matplotlib()

    def reportinfo(self):
        return self.path, 0, self.name


class DoNothingFile(pytest.File):
    """
    Custom file handler to avoid running twice python files explicitly provided on the command line.
    """

    def collect(self):
        return []
