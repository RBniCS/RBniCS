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

import os
from rbnics.backends.dolfin import BasisFunctionsMatrix as DolfinBasisFunctionsMatrix
from rbnics.backends.dolfin.wrapping import get_mpi_comm
from rbnics.utils.decorators import overload
from rbnics.utils.mpi import parallel_io

class NonHierarchicalBasisFunctionsMatrix(object):
    def __init__(self, V):
        self.V = V
        self._components_name = None
        self._content = dict()
        self.mpi_comm = get_mpi_comm(V)
        
    def init(self, components_name):
        self._components_name = components_name
        
    @overload(slice) # e.g. key = :N, return the first N functions
    def __getitem__(self, key):
        N = self._convert_key(key)
        assert N in self._content
        return self._content[N]
        
    @overload(slice, object) # the second argument is object in order to handle FunctionsList's AdditionalFunctionType
    def __setitem__(self, key, item):
        N = self._convert_key(key)
        self._content[N] = item
        
    def _convert_key(self, key):
        assert key.start is None
        assert key.step is None
        assert isinstance(key.stop, (dict, int))
        if isinstance(key.stop, dict):
            assert len(key.stop) == 1
            assert "u" in key.stop
            N = key.stop["u"]
        else:
            N = key.stop
        return N
        
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (N, basis_N) in self._content.items():
            basis_N.save(directory, filename + "_N=" + str(N))
            
    def _save_Nmax(self, directory, filename):
        def save_Nmax_task():
            with open(os.path.join(str(directory), filename + ".length"), "w") as length:
                length.write(str(len(self)))
        parallel_io(save_Nmax_task)
        
    def load(self, directory, filename):
        if len(self._content) > 0: # avoid loading multiple times
            return False
        else:
            Nmax = self._load_Nmax(directory, filename)
            for N in range(1, Nmax + 1):
                self._content[N] = DolfinBasisFunctionsMatrix(self.V)
                self._content[N].init(self._components_name)
                return_value = self._content[N].load(directory, filename + "_N=" + str(N))
                assert return_value
            return True
        
    def _load_Nmax(self, directory, filename):
        def load_Nmax_task():
            with open(os.path.join(str(directory), filename + ".length"), "r") as length:
                return int(length.readline())
        return parallel_io(load_Nmax_task)
        
    def __len__(self):
        if len(self._content) > 0:
            assert min(self._content.keys()) == 1
            assert max(self._content.keys()) == len(self._content)
            return len(self._content)
        else:
            return 0
