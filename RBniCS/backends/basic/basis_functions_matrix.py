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

from RBniCS.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix
from RBniCS.utils.decorators import Extends, override
from RBniCS.utils.mpi import is_io_process

###########################     OFFLINE STAGE     ########################### 
## @defgroup OfflineStage Methods related to the offline stage
#  @{

@Extends(AbstractBasisFunctionsMatrix)
class BasisFunctionsMatrix(AbstractBasisFunctionsMatrix):
    @override
    def __init__(self, V_or_Z, backend, wrapping, online_backend):
        self.V_or_Z = V_or_Z
        self.mpi_comm = wrapping.get_mpi_comm(V_or_Z)
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
        self._components = list() # of FunctionsList
        self._precomputed_slices = dict() # from tuple to FunctionsList

    @override
    def init(self, n_components):
        self.clear()
        for c in range(n_components):
            self._components.append(self.backend.FunctionsList(self.V_or_Z))
            
    @override
    def enrich(self, functions, function_component=None, basis_component=0):
        assert function_component is None or basis_component < len(self._components)
        self._components[basis_component].enrich(functions, function_component)
        
    @override
    def clear(self):
        self._components = list()
        
    @override
    def load(self, directory, filename):
        assert len(self._components) > 0
        if len(self._components) > 1:
            return_value = True
            for (component, basis_functions) in enumerate(self._components):
                return_value_component = basis_functions.load(directory, filename + "_component_" + str(component))
                return_value = return_value and return_value_component
            return return_value
        else:
            return self._components[0].load(directory, filename)
        
    @override
    def save(self, directory, filename):
        if len(self._components) > 1:
            for (component, basis_functions) in enumerate(self._components):
                basis_functions.save(directory, filename + "_component_" + str(component))
        else:
            self._components[0].save(directory, filename)
            
    def _linearize_components(self, N=None):
        if N is None:
            N = tuple([len(basis_functions) for basis_functions in self._components])
        if not N in self._precomputed_slices:
            self._precomputed_slices[N] = self.backend.FunctionsList(self.V_or_Z)
            for (component, N_component) in enumerate(N):
                self._precomputed_slices[N].enrich(self._components[component][:N_component], copy=False)
        return self._precomputed_slices[N]
    
    @override
    def __mul__(self, other):
        linearized_basis_functions_matrix = self._linearize_components()
        return linearized_basis_functions_matrix*other
        
    @override
    def __len__(self):
        assert len(self._components) == 1
        return len(self._components[0])

    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None 
            assert key.step is None
            assert isinstance(key.stop, (int, tuple))
            if isinstance(key.stop, int):
                return self._linearize_components((key.stop, ))
            else:
                return self._linearize_components(key.stop)
        else:
            if len(self._components) == 1: # spare the user an obvious extraction of the first component return basis function number key
                return self._components[0][key]
            else: # return all basis functions for each component, then the user may use __getitem__ of FunctionsList to extract a single basis function
                return self._components[key]
            
    @override
    def __iter__(self):
        linearized_basis_functions_matrix = _linearize_components()
        return linearized_basis_functions_matrix.__iter__()
        
