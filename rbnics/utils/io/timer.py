# Copyright (C) 2015-2019 by the RBniCS authors
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

from timeit import default_timer as python_timer
from mpi4py import MPI
from mpi4py.MPI import MAX, SUM

class Timer(object):
    def __init__(self, mode):
        assert mode in ("serial", "parallel")
        self._mode = mode
        self._start = None
        self._comm = MPI.COMM_WORLD
        
    def start(self):
        self._start = python_timer()
        
    def stop(self):
        elapsed = python_timer() - self._start
        self._start = None
        if self._mode == "serial":
            return self._comm.allreduce(elapsed, op=MAX)
        elif self._mode == "parallel":
            return self._comm.allreduce(elapsed, op=SUM)
        else:
            raise ValueError("Invalid mode for timer")
