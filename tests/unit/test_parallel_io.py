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

import pytest
from mpi4py.MPI import COMM_WORLD
from rbnics.utils.mpi import parallel_io

def test_parallel_io_without_return_value():
    def task():
        pass
    return_value = parallel_io(task)
    assert return_value is None
    
def test_parallel_io_with_return_value():
    def task():
        return COMM_WORLD.rank
    return_value = parallel_io(task)
    assert return_value is 0
    
def test_parallel_io_with_error_1():
    exception_message = "This test will fail"
    def task():
        raise RuntimeError(exception_message)
    with pytest.raises(RuntimeError) as excinfo:
        parallel_io(task)
    assert str(excinfo.value) == exception_message
    
class CustomError(RuntimeError):
    def __init__(self, arg1, arg2):
        RuntimeError.__init__(self, arg1, arg2)
    
def test_parallel_io_with_error_2():
    exception_message_1 = "This test"
    exception_message_2 = "will fail"
    def task():
        raise CustomError(exception_message_1, exception_message_2)
    with pytest.raises(CustomError) as excinfo:
        parallel_io(task)
    assert str(excinfo.value) == str((exception_message_1, exception_message_2))
