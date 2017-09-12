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
from rbnics.utils.decorators import Extends
from rbnics.utils.mpi import is_io_process

# Type for storing a list of functions. From the user point of view this is
# the same as a matrix. Indeed, given a Matrix A, a Vector F 
# and a FunctionsList Z, overriding __mul__ and __rmul__ operators
# allows to write expressions like transpose(Z)*A*Z and transpose(Z)*F
@Extends(AbstractFunctionsList)
class FunctionsList(AbstractFunctionsList):
    def __init__(self, V_or_Z, component, backend, wrapping, AdditionalIsFunction=None, ConvertAdditionalFunctionTypes=None):
        if component is None:
            self.V_or_Z = V_or_Z
        else:
            self.V_or_Z = wrapping.get_function_subspace(V_or_Z, component)
        self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
        self.backend = backend
        self.wrapping = wrapping
        if AdditionalIsFunction is None:
            def _AdditionalIsFunction(arg):
                return False
            self.AdditionalIsFunction = _AdditionalIsFunction
        else:
            self.AdditionalIsFunction = AdditionalIsFunction
        if ConvertAdditionalFunctionTypes is None:
            def _ConvertAdditionalFunctionTypes(arg):
                raise NotImplementedError("Please implement conversion of additional function types")
            self.ConvertAdditionalFunctionTypes = _ConvertAdditionalFunctionTypes
        else:
            self.ConvertAdditionalFunctionTypes = ConvertAdditionalFunctionTypes
        self._list = list() # of functions
        self._precomputed_slices = dict() # from tuple to FunctionsList
    
    def enrich(self, functions, component=None, weights=None, copy=True):
        # Append to storage
        assert isinstance(functions, (tuple, list, FunctionsList, self.backend.Function.Type())) or self.AdditionalIsFunction(functions)
        if isinstance(functions, self.backend.Function.Type()) or self.AdditionalIsFunction(functions):
            if self.AdditionalIsFunction(functions):
                functions = self.ConvertAdditionalFunctionTypes(functions)
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
            assert len(component) == 1
            for (component_from, component_to) in component.items():
                break
            self._list.append(self.wrapping.function_extend_or_restrict(function, component_from, self.V_or_Z, component_to, weight, copy))
        
    def clear(self):
        self._list = list()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (index, function) in enumerate(self._list):
            self.wrapping.function_save(function, directory, filename + "_" + str(index))
                
    def _save_Nmax(self, directory, filename):
        if is_io_process(self.mpi_comm):
            with open(str(directory) + "/" + filename + ".length", "w") as length:
                length.write(str(len(self._list)))
        
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
    
    def __len__(self):
        return len(self._list)

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
            
    def __setitem__(self, key, item):
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        self._list[key] = item
            
    def __iter__(self):
        return self._list.__iter__()
        
