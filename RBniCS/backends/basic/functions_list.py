# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.abstract import FunctionsList as AbstractFunctionsList
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi import is_io_process

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

# Type for storing a list of functions. From the user point of view this is
# the same as a matrix. Indeed, given a Matrix A, a Vector F 
# and a FunctionsList Z, overriding __mul__ and __rmul__ operators
# allows to write expressions like transpose(Z)*A*Z and transpose(Z)*F
@Extends(AbstractFunctionsList)
class FunctionsList(AbstractFunctionsList):
    @override
    def __init__(self, V_or_Z, backend, wrapping, online_backend):
        self.V_or_Z = V_or_Z
        self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
        self._list = list() # of functions
        self._precomputed_slices = dict() # from tuple to FunctionsList
    
    @override
    def enrich(self, functions, component=None, copy=True):
        def append(function):
            if self.wrapping.function_component_as_restriction(function, component, self.V_or_Z):
                self._list.append(self.wrapping.function_component(function, component, copy))
            else: # must be extended from subspace to function space
                self._list.append(self.wrapping.function_extend(function, component, self.V_or_Z))
        # Append to storage
        assert isinstance(functions, (tuple, list, FunctionsList, self.backend.Function.Type()))
        if isinstance(functions, (tuple, list, FunctionsList)):
            for function in functions:
                append(function)
        elif isinstance(functions, self.backend.Function.Type()):
            append(functions)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.enrich.")
        # Reset precomputed slices
        self._precomputed_slices = dict()
        # Prepare trivial precomputed slice
        self._precomputed_slices[len(self._list)] = self
        
    @override
    def clear(self):
        self._list = list()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def save(self, directory, filename):
        self._save_Nmax(directory, filename)
        for (index, fun) in enumerate(self._list):
            self.wrapping.function_save(fun, directory, filename + "_" + str(index))
                
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
            self.enrich(self.wrapping.function_load(directory, filename + "_" + str(index), self.V_or_Z))
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
        assert isinstance(other, (self.online_backend.Matrix.Type(), self.online_backend.Vector.Type(), tuple, self.online_backend.Function.Type()))
        if isinstance(other, self.online_backend.Matrix.Type()):
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_matrix(self, other, self.backend.FunctionsList, self.backend)
        elif isinstance(other, (self.online_backend.Vector.Type(), tuple)): # tuple is used when multiplying by theta_bc
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_vector(self, other, self.backend)
        elif isinstance(other, self.online_backend.Function.Type()):
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
        self._list[key] = key
            
    @override
    def __iter__(self):
        return self._list.__iter__()
        
