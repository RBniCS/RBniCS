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

from dolfin import Function, FunctionSpace
from rbnics.backends.dolfin import BasisFunctionsMatrix as DolfinBasisFunctionsMatrix
from rbnics.utils.decorators import overload
from rbnics.utils.mpi import is_io_process

class NonHierarchicalBasisFunctionsMatrix(DolfinBasisFunctionsMatrix):
    def init(self, components_name):
        assert len(self._components_name) is 0 # re-initialization is not supported here
        # Call Parent
        DolfinBasisFunctionsMatrix.init(self, components_name)
        
    def enrich(self, functions, component=None, weights=None, copy=True):
        raise RuntimeError("Enrichment is not supported here, only slice assignment")
        
    def clear(self):
        raise RuntimeError("Clear is not supported here, only slice assignment")
        
    def _prepare_trivial_precomputed_slice(self):
        raise RuntimeError("This method should have never been called")
        
    @overload(slice, object) # the second argument is object in order to handle FunctionsList's AdditionalFunctionType
    def __setitem__(self, key, item):
        assert key.start is None
        assert key.step is None
        assert isinstance(key.stop, int)
        self._precomputed_slices[key.stop] = item
        
    def _precompute_slice(self, N): # used by __getitem__
        assert isinstance(N, int)
        assert N in self._precomputed_slices
        return self._precomputed_slices[N]
        
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (N, basis_N) in self._precomputed_slices.items():
            basis_N.save(directory, filename + "_N=" + str(N))
            
    def _save_Nmax(self, directory, filename):
        if len(self._precomputed_slices) > 0:
            assert min(self._precomputed_slices.keys()) == 1
            assert max(self._precomputed_slices.keys()) == len(self._precomputed_slices)
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self._precomputed_slices)))
        
    def load(self, directory, filename):
        Nmax = self._load_Nmax(directory, filename)
        return_value = True
        for N in range(1, Nmax + 1):
            self._precomputed_slices[N] = DolfinBasisFunctionsMatrix(self.V_or_Z)
            self._precomputed_slices[N].init(self._components_name)
            return_value_N = self._precomputed_slices[N].load(directory, filename + "_N=" + str(N))
            return_value = return_value and return_value_N
        return return_value
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
