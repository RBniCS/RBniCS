# Copyright (C) 2015-2022 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import sys
import importlib
import pytest
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


def pytest_collect_file(file_path, path, parent):
    """
    Hook into py.test to collect tutorial files.
    """
    if file_path.suffix == ".py" and file_path.name.startswith("tutorial_"):
        return TutorialFile.from_parent(parent=parent, path=file_path)


def pytest_pycollect_makemodule(module_path, path, parent):
    """
    Hook into py.test to avoid collecting twice tutorial files explicitly provided on the command lines
    """
    if module_path.suffix == ".py" and module_path.name.startswith("tutorial_"):
        return DoNothingFile.from_parent(parent=parent, path=module_path)


class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files
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
    Custom file handler to avoid running twice tutorial files explicitly provided on the command lines
    """

    def collect(self):
        return []
