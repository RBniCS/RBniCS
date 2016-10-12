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
        self._len_components = dict() # of int
        self._basis_component_index_to_component_name = dict() # filled in by init
        self._component_name_to_basis_component_index = dict() # filled in by init
        self._component_name_to_function_component = dict() # filled in by init

    @override
    def init(self, component_name_to_basis_component_index, component_name_to_function_component):
        if not (
            self._component_name_to_basis_component_index == component_name_to_basis_component_index
                and
            self._component_name_to_function_component == component_name_to_function_component
        ): # Do nothing if it was already initialize with the same dicts
            self.clear()
            # Initialize components FunctionsList
            for component_name in component_name_to_basis_component_index:
                self._components[component_name] = self.backend.FunctionsList(self.V_or_Z)
            # Store the component_name_to_function_component dict
            self._component_name_to_function_component = component_name_to_function_component
            # Store the component_name_to_basis_component_index dict
            self._component_name_to_basis_component_index = component_name_to_basis_component_index
            assert max(component_name_to_basis_component_index.values()) == len(component_name_to_basis_component_index) - 1
            # Reverse the component_name_to_basis_component_index dict and store it
            self._basis_component_index_to_component_name = dict()
            for (component_name, basis_component_index) in component_name_to_basis_component_index.iteritems():
                self._basis_component_index_to_component_name[basis_component_index] = component_name
            # Prepare len components
            self._len_components = dict()
            for component_name in component_name_to_basis_component_index:
                self._len_components[component_name] = 0
            
    @override
    def enrich(self, functions, component_name=None, copy=True):
        assert component_name is None or component_name in self._components
        assert copy is True
        if component_name is None:
            assert len(self._components) == 1
            component_0 = self._components.keys()[0]
            self._components[component_0].enrich(functions)
            self._len_components[component_0] = len(self._components[component_0])
        else:
            self._components[component_name].enrich(functions, self._component_name_to_function_component[component_name])
            self._len_components[component_name] = len(self._components[component_name])
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def clear(self):
        self._components = dict()
        self._len_components = dict()
        # Reset precomputed slices
        self._precomputed_slices = dict()
        
    @override
    def load(self, directory, filename):
        assert len(self._components) > 0
        if len(self._components) > 1:
            return_value = True
            for (component_name, basis_functions) in self._components.iteritems():
                return_value_component = basis_functions.load(directory, filename + "_component_" + component_name)
                return_value = return_value and return_value_component
                # Also populate component length
                self._len_components[component_name] = len(basis_functions)
            return return_value
        else:
            component_0 = self._components.keys()[0]
            return_value = self._components[component_0].load(directory, filename)
            # Also populate component length
            self._len_components[component_0] = len(self._components[component_0])
            # Return
            return return_value
        
    @override
    def save(self, directory, filename):
        if len(self._components) > 1:
            for (component_name, basis_functions) in self._components.iteritems():
                basis_functions.save(directory, filename + "_component_" + component_name)
        else:
            component_0 = self._components.keys()[0]
            self._components[component_0].save(directory, filename)
        
    @override
    def __mul__(self, other):
        assert isinstance(other, (self.online_backend.Matrix.Type(), self.online_backend.Vector.Type(), tuple, self.online_backend.Function.Type()))
        if isinstance(other, self.online_backend.Matrix.Type()):
            def BasisFunctionsMatrixWithInit(V_or_Z):
                output = self.backend.BasisFunctionsMatrix(V_or_Z)
                output.init(self._component_name_to_basis_component_index, self._component_name_to_function_component)
                return output
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_matrix(self, other, BasisFunctionsMatrixWithInit, self.backend)
        elif isinstance(other, (self.online_backend.Vector.Type(), tuple)): # tuple is used when multiplying by theta_bc
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_vector(self, other, self.backend)
        elif isinstance(other, self.online_backend.Function.Type()):
            return self.wrapping.functions_list_basis_functions_matrix_mul_online_function(self, other, self.backend)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList.__mul__.")
        
    @override
    def __len__(self):
        assert len(self._len_components) == 1
        component_0 = self._len_components.keys()[0]
        return self._len_components[component_0]

    @override
    def __getitem__(self, key):
        if isinstance(key, slice): # e.g. key = :N, return the first N functions
            assert key.start is None 
            assert key.step is None
            assert isinstance(key.stop, (int, dict))
            return self._precompute_slice(key.stop)
        else:
            if len(self._components) == 1: # spare the user an obvious extraction of the first component return basis function number key
                component_0 = self._components.keys()[0]
                return self._components[component_0][key]
            else: # return all basis functions for each component, then the user may use __getitem__ of FunctionsList to extract a single basis function
                return self._components[key]
            
    def _precompute_slice(self, N=None):
        assert isinstance(N, (int, dict))
        if isinstance(N, dict):
            N_key = tuple(N.iteritems()) # convert dict to a hashable type
        else:
            assert len(self._components) == 1
            N_key = N
        if not N_key in self._precomputed_slices:
            self._precomputed_slices[N_key] = self.backend.BasisFunctionsMatrix(self.V_or_Z)
            self._precomputed_slices[N_key].init(self._component_name_to_basis_component_index, self._component_name_to_function_component)
            for (basis_component_index, component_name) in sorted(self._basis_component_index_to_component_name.iteritems()):
                if isinstance(N, dict):
                    N_component = N[component_name]
                else:
                    N_component = N
                self._precomputed_slices[N_key]._components[component_name].enrich(self._components[component_name][:N_component], copy=False)
                self._precomputed_slices[N_key]._len_components[component_name] = len(self._precomputed_slices[N_key]._components[component_name])
        return self._precomputed_slices[N_key]
        
    @override
    def __iter__(self):
        raise NotImplementedError("BasisFunctionsMatrix.iter() has not been implemented yet")
        
