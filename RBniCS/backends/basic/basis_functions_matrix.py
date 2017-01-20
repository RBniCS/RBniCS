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
        self._components_name = list() # filled in by init
        self._component_name_to_basis_component_index = dict() # filled in by init
        self._component_name_to_basis_component_length = dict() # of int

    def init(self, components_name):
        if self._components_name != components_name: # Do nothing if it was already initialize with the same dicts
            # Store components name
            self._components_name = components_name
            # Initialize components FunctionsList
            self._components.clear()
            for component_name in components_name:
                self._components[component_name] = self.backend.FunctionsList(self.V_or_Z)
            # Intialize the component_name_to_basis_component_index dict, and its inverse
            self._component_name_to_basis_component_index.clear()
            self._basis_component_index_to_component_name.clear()
            for (basis_component_index, component_name) in enumerate(components_name):
                self._component_name_to_basis_component_index[component_name] = basis_component_index
                self._basis_component_index_to_component_name[basis_component_index] = component_name
            # Prepare len components
            self._component_name_to_basis_component_length.clear()
            for component_name in self._components_name:
                self._component_name_to_basis_component_length[component_name] = 0
            # Reset precomputed slices
            self._precomputed_slices.clear()
            
    @override
    def enrich(self, functions, component=None, weights=None, copy=True):
        assert copy is True
        if component is None:
            assert len(self._components) == 1
            component_0 = self._components.keys()[0]
            self._components[component_0].enrich(functions, weights=weights)
            self._component_name_to_basis_component_length[component_0] = len(self._components[component_0])
        else:
            assert isinstance(component, (str, dict))
            if isinstance(component, str):
                assert component in self._components
                self._components[component].enrich(functions, component=component, weights=weights)
                self._component_name_to_basis_component_length[component] = len(self._components[component])
            else:
                assert len(component.values()) == 1
                component_to = component.values()[0]
                assert component_to in self._components
                self._components[component_to].enrich(functions, component=component, weights=weights)
                self._component_name_to_basis_component_length[component_to] = len(self._components[component_to])
        # Reset and prepare precomputed slices
        self._prepare_trivial_precomputed_slice()
            
    def _prepare_trivial_precomputed_slice(self):
        # Reset precomputed slices
        self._precomputed_slices.clear()
        # Prepare trivial precomputed slice
        if len(self._components) == 1:
            component_0 = self._components.keys()[0]
            precomputed_slice_key = self._component_name_to_basis_component_length[component_0]
        else:
            precomputed_slice_key = list()
            for (basis_component_index, component_name) in sorted(self._basis_component_index_to_component_name.iteritems()):
                precomputed_slice_key.append(self._component_name_to_basis_component_length[component_name])
            precomputed_slice_key = tuple(precomputed_slice_key)
        self._precomputed_slices[precomputed_slice_key] = self
        
    @override
    def clear(self):
        components_name = self._components_name
        # Trick _init into re-initializing everything
        self._components_name = None
        self.init(components_name)
        
    @override
    def save(self, directory, filename):
        if len(self._components) > 1:
            for (component_name, basis_functions) in self._components.iteritems():
                basis_functions.save(directory, filename + "_component_" + component_name)
        else:
            component_0 = self._components.keys()[0]
            self._components[component_0].save(directory, filename)
        
    @override
    def load(self, directory, filename):
        return_value = True
        assert len(self._components) > 0
        if len(self._components) > 1:
            for (component_name, basis_functions) in self._components.iteritems():
                return_value_component = basis_functions.load(directory, filename + "_component_" + component_name)
                return_value = return_value and return_value_component
                # Also populate component length
                self._component_name_to_basis_component_length[component_name] = len(basis_functions)
        else:
            component_0 = self._components.keys()[0]
            return_value = self._components[component_0].load(directory, filename)
            # Also populate component length
            self._component_name_to_basis_component_length[component_0] = len(self._components[component_0])
        # Reset and prepare precomputed slices
        self._prepare_trivial_precomputed_slice()
        # Return
        return return_value
        
    @override
    def __mul__(self, other):
        assert isinstance(other, (self.online_backend.Matrix.Type(), self.online_backend.Vector.Type(), tuple, self.online_backend.Function.Type()))
        if isinstance(other, self.online_backend.Matrix.Type()):
            def BasisFunctionsMatrixWithInit(V_or_Z):
                output = self.backend.BasisFunctionsMatrix(V_or_Z)
                output.init(self._components_name)
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
        assert len(self._component_name_to_basis_component_length) == 1
        component_0 = self._component_name_to_basis_component_length.keys()[0]
        return self._component_name_to_basis_component_length[component_0]

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
                
    @override
    def __setitem__(self, key, item):
        assert not isinstance(key, slice) # only able to set the element at position "key" in the storage
        assert len(self._components) == 1, "Cannot set components, only single functions. Did you mean to call __getitem__ to extract a component and __setitem__ of a single function on that component?"
        component_0 = self._components.keys()[0]
        self._components[component_0][key] = item
            
    def _precompute_slice(self, N):
        assert isinstance(N, (int, dict))
        if isinstance(N, dict):
            N_key = list()
            for (basis_component_index, component_name) in sorted(self._basis_component_index_to_component_name.iteritems()):
                N_key.append(N[component_name])
            N_key = tuple(N_key)
        else:
            assert len(self._components) == 1
            N_key = N
        if not N_key in self._precomputed_slices:
            self._precomputed_slices[N_key] = self.backend.BasisFunctionsMatrix(self.V_or_Z)
            self._precomputed_slices[N_key].init(self._components_name)
            for (basis_component_index, component_name) in sorted(self._basis_component_index_to_component_name.iteritems()):
                if isinstance(N, dict):
                    N_component = N[component_name]
                else:
                    N_component = N
                self._precomputed_slices[N_key]._components[component_name].enrich(self._components[component_name][:N_component], copy=False)
                self._precomputed_slices[N_key]._component_name_to_basis_component_length[component_name] = len(self._precomputed_slices[N_key]._components[component_name])
        return self._precomputed_slices[N_key]
        
    @override
    def __iter__(self):
        raise NotImplementedError("BasisFunctionsMatrix.iter() has not been implemented yet")
        
