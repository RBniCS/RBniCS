# Copyright (C) 2015-2018 by the RBniCS authors
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
import types
import pytest
import dolfin # otherwise the next import from rbnics would disable dolfin as a required backend  # noqa
from rbnics.utils.test import add_gold_options, disable_matplotlib, enable_matplotlib, process_gold_options, run_and_compare_to_gold

def pytest_addoption(parser):
    add_gold_options(parser, "RBniCS")
    
def pytest_configure(config):
    process_gold_options(config)

def pytest_pycollect_makemodule(path, parent):
    """
    Hook into py.test to collect test files.
    """
    assert path.ext == ".py"
    assert path.basename.startswith("test_")
    return TestFile(path, parent)
    
class TestFile(pytest.Module):
    """
    Custom file handler for test files
    """
    
    def makeitem(self, name, obj):
        test_item = pytest.Module.makeitem(self, name, obj)
        assert test_item is None or isinstance(test_item, list)
        if test_item is None:
            return None
        else:
            for test_item_i in test_item:
                patch_test_item(test_item_i)
            return test_item
        
def patch_test_item(test_item):
    """
    Handle the execution of the test.
    """
    
    subdirectory = os.path.join(test_item.originalname + "_tempdir", test_item.callspec.getparam("expression_type"), test_item.callspec.getparam("basis_generation"))
    original_runtest = test_item.runtest
    
    @run_and_compare_to_gold(subdirectory)
    def runtest(self):
        disable_matplotlib()
        original_runtest()
        enable_matplotlib()
        
    test_item.runtest = types.MethodType(runtest, test_item)
