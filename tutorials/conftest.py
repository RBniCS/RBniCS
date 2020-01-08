# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

import os
import sys
import importlib
import pytest
import dolfin # otherwise the next import from rbnics would disable dolfin as a required backend  # noqa: F401
from rbnics.utils.test import add_gold_options, disable_matplotlib, enable_matplotlib, process_gold_options, run_and_compare_to_gold

def pytest_addoption(parser):
    add_gold_options(parser, "RBniCS")
    
def pytest_configure(config):
    process_gold_options(config)

def pytest_collect_file(path, parent):
    """
    Hook into py.test to collect tutorial files.
    """
    if path.ext == ".py" and path.basename.startswith("tutorial_"):
        return TutorialFile(path, parent)
        
def pytest_pycollect_makemodule(path, parent):
    """
    Hook into py.test to avoid collecting twice tutorial files explicitly provided on the command lines
    """
    if path.ext == ".py" and path.basename.startswith("tutorial_"):
        return DoNothingFile(path, parent)
        
class TutorialFile(pytest.File):
    """
    Custom file handler for tutorial files
    """
    
    def collect(self):
        yield TutorialItem(os.path.relpath(str(self.fspath), str(self.parent.fspath)), self)
        
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
    Custom file handler to avoid running twice tutorial files explicitly provided on the command lines
    """
    
    def collect(self):
        return []
