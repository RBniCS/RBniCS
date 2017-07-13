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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
import rbnics.backends.online
from rbnics.utils.decorators import Extends, override
from rbnics.utils.mpi import is_io_process

# Type for storing a list of functions. From the user point of view this is
# the same as a matrix. Indeed, given a Matrix A, a Vector F 
# and a FunctionsList Z, overriding __mul__ and __rmul__ operators
# allows to write expressions like transpose(Z)*A*Z and transpose(Z)*F
@Extends(AbstractFunctionsList)
class FunctionsList(AbstractFunctionsList):
    @override
    def __init__(self, V_or_Z, component, backend, wrapping, AdditionalFunctionTypes=None):
        if component is None:
            self.V_or_Z = V_or_Z
        else:
            self.V_or_Z = wrapping.get_function_subspace(V_or_Z, component)
        self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
        self.backend = backend
        self.wrapping = wrapping
        if AdditionalFunctionTypes is None:
            self.FunctionTypes = (backend.Function.Type(), )
        else:
            self.FunctionTypes = AdditionalFunctionTypes + (backend.Function.Type(), )
        self._list = list() # of functions
        self._precomputed_slices = dict() # from tuple to FunctionsList
    
    @override
    def enrich(self, functions, component=None, weights=None, copy=True):
        # Append to storage
        assert isinstance(functions, (tuple, list, FunctionsList, ) + self.FunctionTypes)
        if isinstance(functions, self.FunctionTypes):
            self._enrich(functions, component=component, weight=weights, copy=copy)
        elif isinstance(functions, (tuple, list, FunctionsList)):
            if weights is not None:
                assert len(weights) == len(functions)
                for (index, function) in enumerate(functions):
                    self._enrich(function, component=component, weight=weights[index], copy=copy)
            else:
                for function in functions:
                    self._enrich(function, component=component, copy=copy)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.enrich.")
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        self._precomputed_slices[len(self._list)] = self
        
    def _enrich(self, function, component=None, weight=None, copy=True):
        assert component is None or isinstance(component, (str, dict))
        if component is None or isinstance(component, str):
            self._list.append(self.wrapping.function_extend_or_restrict(function, component, self.V_or_Z, component, weight, copy))
        else:
            assert len(component.keys()) == 1
            component_from = component.keys()[0]
            assert len(component.values()) == 1
            component_to = component.values()[0]
            self._list.append(self.wrapping.function_extend_or_restrict(function, component_from, self.V_or_Z, component_to, weight, copy))
        
    @override
    def clear(self):
        self._list = list()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (index, function) in enumerate(self._list):
            self.wrapping.function_save(function, directory, filename + "_" + str(index))
                
    def _save_Nmax(self, directory, filename):
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self._list)))
        
    @override
    def load(self, directory, filename):
        if len(self._list) > 0: # avoid loading multiple times
            return False
        Nmax = self._load_Nmax(directory, filename)
        for index in range(Nmax):
            function = self.backend.Function(self.V_or_Z)
            loaded = self.wrapping.function_load(function, directory, filename + "_" + str(index))
            assert loaded
            self.enrich(function)
        return True
        
    def _load_Nmax(self, directory, filename):
        Nmax = None
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "r") as length:
                Nmax = int(length.readline())
        Nmax = self.mpi_comm.bcast(Nmax, root=is_io_process.root)
        return Nmax
    
    @override
    def __mul__(self, other):
        assert isinstance(other, (rbnics.backends.online.OnlineMatrix.Type(), rbnics.backends.online.OnlineVector.Type(), tuple, rbnics.backends.online.OnlineFunction.Type()))
        if isinstance(other, rbnics.backends.online.OnlineMatrix.Type()):
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_matrix(self, other, self.backend.FunctionsList, self.backend)
        elif isinstance(other, (rbnics.backends.online.OnlineVector.Type(), tuple)): # tuple is used when multiplying by theta_bc
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_vector(self, other, self.backend)
        elif isinstance(other, rbnics.backends.online.OnlineFunction.Type()):
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_function(self, other, self.backend)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.__mul__.")
    
    @override
    def __len__(self):
        return len(self._list)

    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None 
            assert key.step is None
            assert key.stop <= len(self._list)
            
            if key.stop in self._precomputed_slices:
                return self._precomputed_slices[key.stop]
            else:
                output = self.backend.FunctionsList(self.V_or_Z)
                output._list = self._list[key]
                self._precomputed_slices[key.stop] = output
                return output
                
        else: # return the element at position "key" in the storage
            return self._list[key]
            
    @override
    def __setitem__(self, key, item):
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._list[key] = item
            
    @override
    def __iter__(self):
        return self._list.__iter__()
        
