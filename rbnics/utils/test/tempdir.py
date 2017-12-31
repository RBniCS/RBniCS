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
import shutil
import pytest
from dolfin_utils.test import tempdir

# Temporarily change the tempdir fixture to avoid it clearing out the temporary folder
os_mkdir = os.mkdir
shutil_rmtree = shutil.rmtree

def do_not_rmtree(arg):
    pass

def mkdir_for_save(arg):
    os_mkdir(arg.replace("_test_tensor_save", "test_tensor_io"))
    
def mkdir_for_load(arg):
    pass
    
@pytest.fixture(scope="function")
def save_tempdir(request):
    function_name = request.function.__name__
    request.function.__name__ = function_name.replace("_save", "_io")
    os.mkdir = mkdir_for_save
    output = tempdir(request)
    request.function.__name__ = function_name
    os.mkdir = os_mkdir
    return output

@pytest.fixture(scope="function")
def load_tempdir(request):
    function_name = request.function.__name__
    request.function.__name__ = function_name.replace("_load", "_io")
    os.mkdir = mkdir_for_load
    shutil.rmtree = do_not_rmtree
    output = tempdir(request)
    request.function.__name__ = function_name
    os.mkdir = os_mkdir
    shutil.rmtree = shutil_rmtree
    return output
