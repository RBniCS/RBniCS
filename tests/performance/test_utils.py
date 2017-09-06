# Copyright (C) 2015-2017 by the RBniCS authors
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

import sys
import timeit
from dolfin import Function
from numpy import mean, zeros
from numpy.random import random, randint
from rbnics.backends.online.numpy import Matrix as NumpyMatrix, Vector as NumpyVector

class TestBase(object):
    def __init__(self):
        self.test_id = 0
        self.test_subid = None
        #
        self.index = 0
        self.storage = {}
        #
        self.timer = _Timer(self)
            
    def init_test(self, test_id, test_subid=None):
        self.test_id = test_id
        self.test_subid = test_subid
    
    def run(self):
        pass
    
    def timeit(self):
        def run_timeit(self):
            self.index = 0
            r = self.timer.timeit_timer.repeat(repeat=3, number=self.timer.number)
            return min(r) * 1e6 / self.timer.number # usec
        if self.test_id is 0:
            # Need to run timeit twice: the first one to populate the storage,
            # the second one to time access to the storage
            usec_build = run_timeit(self)
            usec_access = run_timeit(self)
            return (usec_build, usec_access)
        else:
            return run_timeit(self)
        
    def average(self):
        self.index = 0
        total_times = 3*self.timer.number
        all_outputs = zeros((total_times))
        for i in range(total_times):
            all_outputs[i] = self.run()
        return mean(all_outputs)
    
    def number_of_runs(self):
        return self.timer.number

def RandomDolfinFunction(V):
    f = Function(V)
    f.vector().set_local(_rand(f.vector().array().size))
    f.vector().apply("insert")
    return f

def RandomNumber():
    return _rand(1)[0]

def RandomNumpyMatrix(N, M):
    m = NumpyMatrix(N, M)
    m[:, :] = _rand(N, M)
    return m
    
def RandomNumpyVector(N):
    v = NumpyVector(N)
    v[:] = _rand(N, 1)
    return v
    
def RandomTuple(Q):
    return tuple(_rand(Q))
        
class _Timer(object):
    def __init__(self, test):
        self.timeit_timer = timeit.Timer(lambda: test.run())
        # Inspired by python's cpython/master/Lib/timeit.py,
        # which is called by python -mtimeit script,
        # determine number so that 0.2 <= total time < 2.0
        for i in range(1, 10):
            test.index = 0
            test.storage = {}
            self.number = 10**i
            r = self.timeit_timer.timeit(self.number)
            if r >= 0.2:
                # Clean up, we do want to time again
                # the construction phase
                test.index = 0
                test.storage = {}
                #
                break
                
def _rand(*args):
    return (-1)**randint(2, size=(args))*random(args)/(1e-3 + random(args))
