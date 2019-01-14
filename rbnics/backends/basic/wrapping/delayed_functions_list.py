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

from rbnics.backends.basic.wrapping.delayed_linear_solver import DelayedLinearSolver
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import overload
from rbnics.utils.io import TextIO as LengthIO

class DelayedFunctionsList(object):
    def __init__(self, space):
        self.space = space
        self._enrich_memory = list()
        self._precomputed_slices = Cache() # from tuple to DelayedFunctionsList
        
    def enrich(self, function, component=None, weight=None, copy=True):
        assert component is None
        assert weight is None
        assert copy is True
        # Append to storage
        self._enrich(function)
        # Reset precomputed slices
        self._precomputed_slices.clear()
        # Prepare trivial precomputed slice
        self._precomputed_slices[0, len(self._enrich_memory)] = self
        
    @overload(DelayedLinearSolver)
    def _enrich(self, function):
        self._enrich_memory.append(function)
        
    @overload(lambda cls: cls)
    def _enrich(self, other):
        assert self.space is other.space
        self._enrich_memory.extend(other._enrich_memory)
        
    @overload(int)
    def __getitem__(self, key):
        return self._enrich_memory[key]
        
    @overload(slice) # e.g. key = :N, return the first N functions
    def __getitem__(self, key):
        if key.start is not None:
            start = key.start
        else:
            start = 0
        assert key.step is None
        if key.stop is not None:
            stop = key.stop
        else:
            stop = len(self._list)
            
        assert start <= stop
        if start < stop:
            assert start >= 0
            assert start < len(self._list)
            assert stop > 0
            assert stop <= len(self._list)
        # elif start == stop
        #    trivial case which will result in an empty FunctionsList
        
        if (start, stop) not in self._precomputed_slices:
            output = DelayedFunctionsList(self.space)
            if start < stop:
                output._enrich_memory = self._enrich_memory[key]
            self._precomputed_slices[start, stop] = output
        return self._precomputed_slices[start, stop]
        
    def __len__(self):
        return len(self._enrich_memory)
        
    def save(self, directory, filename):
        LengthIO.save_file(len(self._enrich_memory), directory, filename + "_length")
        for (index, memory) in enumerate(self._enrich_memory):
            memory.save(directory, filename + "_" + str(index))
        
    def load(self, directory, filename):
        if len(self._enrich_memory) > 0: # avoid loading multiple times
            return False
        else:
            assert LengthIO.exists_file(directory, filename + "_length")
            len_memory = LengthIO.load_file(directory, filename + "_length")
            for index in range(len_memory):
                memory = DelayedLinearSolver()
                memory.load(directory, filename + "_" + str(index))
                self.enrich(memory)
            return True
            
    def get_problem_name(self):
        problem_name = None
        for memory in self._enrich_memory:
            if problem_name is None:
                problem_name = memory.get_problem_name()
            else:
                assert memory.get_problem_name() == problem_name
        return problem_name
