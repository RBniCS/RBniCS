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
        self._components = dict() # of FunctionsList
        self._precomputed_slices = dict() # from tuple to FunctionsList
        self._basis_component_index_to_component_name = dict() # filled in by init
        self._component_name_to_function_component = dict() # filled in by init

    @override
    def init(self, component_name_to_basis_component_index, component_name_to_function_component):
        self.clear()
        self._component_name_to_function_component = component_name_to_function_component
        assert max(component_name_to_basis_component_index.values()) == len(component_name_to_basis_component_index) - 1
        for component_name in component_name_to_basis_component_index:
            self._components[component_name] = self.backend.FunctionsList(self.V_or_Z)
        # Reverse the component_name_to_basis_component_index dict and store it
        self._basis_component_index_to_component_name = dict()
        for (component_name, basis_component_index) in component_name_to_basis_component_index.iteritems():
            self._basis_component_index_to_component_name[basis_component_index] = component_name
            
    @override
    def enrich(self, functions, component_name=None, copy=True):
        assert component_name is None or component_name in self._components
        if component_name is None:
            assert len(self._components) == 1
            self._components.values()[0].enrich(functions)
        else:
            self._components[component_name].enrich(functions, self._component_name_to_function_component[component_name])
        
    @override
    def clear(self):
        self._components = dict()
        
    @override
    def load(self, directory, filename):
        assert len(self._components) > 0
        if len(self._components) > 1:
            return_value = True
            for (component_name, basis_functions) in self._components.iteritems():
                return_value_component = basis_functions.load(directory, filename + "_component_" + component_name)
                return_value = return_value and return_value_component
            return return_value
        else:
            return self._components.values()[0].load(directory, filename)
        
    @override
    def save(self, directory, filename):
        if len(self._components) > 1:
            for (component_name, basis_functions) in self._components.iteritems():
                basis_functions.save(directory, filename + "_component_" + component_name)
        else:
            self._components.values()[0].save(directory, filename)
            
    def _linearize_components(self, N=None):
        if N is None:
            N = dict([(component_name, len(basis_functions)) for (component_name, basis_functions) in self._components.iteritems()])
        N_key = tuple(N.iteritems()) # convert dict to a hashable type
        if not N_key in self._precomputed_slices:
            self._precomputed_slices[N_key] = self.backend.FunctionsList(self.V_or_Z)
            for (basis_component_index, component_name) in sorted(self._basis_component_index_to_component_name.iteritems()):
                N_component = N[component_name]
                self._precomputed_slices[N_key].enrich(self._components[component_name][:N_component], copy=False)
        return self._precomputed_slices[N_key]
    
    @override
    def __mul__(self, other):
        linearized_basis_functions_matrix = self._linearize_components()
        return linearized_basis_functions_matrix*other
        
    @override
    def __len__(self):
        assert len(self._components) == 1
        return len(self._components.values()[0])

    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None 
            assert key.step is None
            assert isinstance(key.stop, (int, dict))
            if isinstance(key.stop, int):
                assert len(self._components) == 1
                return self._linearize_components({self._components.keys()[0]: key.stop})
            else:
                return self._linearize_components(key.stop)
        else:
            if len(self._components) == 1: # spare the user an obvious extraction of the first component return basis function number key
                return self._components.values()[0][key]
            else: # return all basis functions for each component, then the user may use __getitem__ of FunctionsList to extract a single basis function
                return self._components[key]
            
    @override
    def __iter__(self):
        linearized_basis_functions_matrix = self._linearize_components()
        return linearized_basis_functions_matrix.__iter__()
        
