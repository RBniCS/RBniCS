# Copyright (C) 2015-2018 by the RBniCS authors
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

from numbers import Number
from rbnics.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix
from rbnics.utils.decorators import dict_of, list_of, overload, ThetaType
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict
from rbnics.utils.test import PatchInstanceMethod

def BasisFunctionsMatrix(backend, wrapping, online_backend, online_wrapping):
    class _BasisFunctionsMatrix(AbstractBasisFunctionsMatrix):
        def __init__(self, space, component=None):
            if component is not None:
                self.space = wrapping.get_function_subspace(space, component)
            else:
                self.space = space
            self.mpi_comm = wrapping.get_mpi_comm(space)
            self._components = dict() # of FunctionsList
            self._precomputed_slices = dict() # from tuple to FunctionsList
            self._components_name = list() # filled in by init
            self._component_name_to_basis_component_index = ComponentNameToBasisComponentIndexDict() # filled in by init
            self._component_name_to_basis_component_length = OnlineSizeDict()

        def init(self, components_name):
            if self._components_name != components_name: # Do nothing if it was already initialized with the same dicts
                # Store components name
                self._components_name = components_name
                # Initialize components FunctionsList
                self._components.clear()
                for component_name in components_name:
                    self._components[component_name] = backend.FunctionsList(self.space)
                # Prepare len components
                self._component_name_to_basis_component_length.clear()
                for component_name in components_name:
                    self._component_name_to_basis_component_length[component_name] = 0
                # Intialize the component_name_to_basis_component_index dict
                self._component_name_to_basis_component_index.clear()
                for (basis_component_index, component_name) in enumerate(components_name):
                    self._component_name_to_basis_component_index[component_name] = basis_component_index
                # Reset precomputed slices
                self._precomputed_slices.clear()
                # Patch FunctionsList.enrich() to update internal attributes
                def patch_functions_list_enrich(component_name, basis_functions):
                    original_functions_list_enrich = basis_functions.enrich
                    def patched_functions_list_enrich(self_, functions, component=None, weights=None, copy=True):
                        # Append to storage
                        original_functions_list_enrich(functions, component, weights, copy)
                        # Update component name to basis component length
                        self._update_component_name_to_basis_component_length(component_name if component is None else component)
                        # Reset precomputed slices
                        self._precomputed_slices.clear()
                        # Prepare trivial precomputed slice
                        self._prepare_trivial_precomputed_slice()
                    basis_functions.enrich_patch = PatchInstanceMethod(basis_functions, "enrich", patched_functions_list_enrich)
                    basis_functions.enrich_patch.patch()
                for component_name in components_name:
                    patch_functions_list_enrich(component_name, self._components[component_name])
                
        def enrich(self, functions, component=None, weights=None, copy=True):
            assert copy is True
            # Append to storage
            self._enrich(functions, component, weights, copy)
        
        @overload(object, None, (None, list_of(Number)), bool) # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        def _enrich(self, functions, component, weights, copy):
            assert len(self._components) == 1
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            self._components[component_0].enrich(functions, None, weights, copy)
            
        @overload(object, str, (None, list_of(Number)), bool) # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        def _enrich(self, functions, component, weights, copy):
            assert component in self._components
            self._components[component].enrich(functions, component, weights, copy)
            
        @overload(object, dict_of(str, str), (None, list_of(Number)), bool) # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        def _enrich(self, functions, component, weights, copy):
            assert len(component) == 1
            for (_, component_to) in component.items():
                break
            assert component_to in self._components
            self._components[component_to].enrich(functions, component, weights)
            
        @overload(None)
        def _update_component_name_to_basis_component_length(self, component):
            assert len(self._components) == 1
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            self._component_name_to_basis_component_length[component_0] = len(self._components[component_0])
            
        @overload(str)
        def _update_component_name_to_basis_component_length(self, component):
            self._component_name_to_basis_component_length[component] = len(self._components[component])
            
        @overload(dict_of(str, str))
        def _update_component_name_to_basis_component_length(self, component):
            assert len(component) == 1
            for (_, component_to) in component.items():
                break
            assert component_to in self._components
            self._component_name_to_basis_component_length[component_to] = len(self._components[component_to])
            
        def _prepare_trivial_precomputed_slice(self):
            if len(self._components) == 1:
                assert len(self._components_name) == 1
                component_0 = self._components_name[0]
                precomputed_slice_key = self._component_name_to_basis_component_length[component_0]
            else:
                precomputed_slice_key = list()
                for component_name in self._components_name:
                    precomputed_slice_key.append(self._component_name_to_basis_component_length[component_name])
                precomputed_slice_key = tuple(precomputed_slice_key)
            self._precomputed_slices[precomputed_slice_key] = self
            
        def clear(self):
            components_name = self._components_name
            # Trick _init into re-initializing everything
            self._components_name = None
            self.init(components_name)
            
        def save(self, directory, filename):
            if len(self._components) > 1:
                def filename_and_component(component_name):
                    return filename + "_" + component_name
            else:
                def filename_and_component(component_name):
                    return filename
            for (component_name, basis_functions) in self._components.items():
                basis_functions.save(directory, filename_and_component(component_name))
            
        def load(self, directory, filename):
            return_value = True
            assert len(self._components) > 0
            if len(self._components) > 1:
                def filename_and_component(component_name):
                    return filename + "_" + component_name
            else:
                def filename_and_component(component_name):
                    return filename
            for (component_name, basis_functions) in self._components.items():
                # Skip updating internal attributes while reading in basis functions, we will do that
                # only once at the end
                assert hasattr(basis_functions, "enrich_patch")
                basis_functions.enrich_patch.unpatch()
                # Load each component
                return_value_component = basis_functions.load(directory, filename_and_component(component_name))
                return_value = return_value and return_value_component
                # Populate component length
                self._update_component_name_to_basis_component_length(component_name)
                # Restore patched enrich method
                basis_functions.enrich_patch.patch()
            # Reset precomputed slices
            self._precomputed_slices.clear()
            # Prepare trivial precomputed slice
            self._prepare_trivial_precomputed_slice()
            # Return
            return return_value
            
        @overload(online_backend.OnlineMatrix.Type(), )
        def __mul__(self, other):
            if isinstance(other.M, dict):
                assert set(other.M.keys()) == set(self._components_name)
            def BasisFunctionsMatrixWithInit(space):
                output = _BasisFunctionsMatrix.__new__(type(self), space)
                output.__init__(space)
                output.init(self._components_name)
                return output
            return wrapping.basis_functions_matrix_mul_online_matrix(self, other, BasisFunctionsMatrixWithInit)
            
        @overload(online_backend.OnlineFunction.Type(), )
        def __mul__(self, other):
            return self.__mul__(online_wrapping.function_to_vector(other))
            
        @overload(online_backend.OnlineVector.Type(), )
        def __mul__(self, other):
            if isinstance(other.N, dict):
                assert set(other.N.keys()) == set(self._components_name)
            return wrapping.basis_functions_matrix_mul_online_vector(self, other)
            
        @overload(ThetaType, )
        def __mul__(self, other):
            return wrapping.basis_functions_matrix_mul_online_vector(self, other)
            
        def __len__(self):
            assert len(self._components_name) == 1
            assert len(self._component_name_to_basis_component_length) == 1
            return self._component_name_to_basis_component_length[self._components_name[0]]
            
        @overload(int)
        def __getitem__(self, key):
            # spare the user an obvious extraction of the first component return basis function number key
            assert len(self._components) == 1
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            return self._components[component_0][key]
                
        @overload(str)
        def __getitem__(self, key):
            # return all basis functions for each component, then the user may use __getitem__ of FunctionsList to extract a single basis function
            return self._components[key]
            
        @overload(slice) # e.g. key = :N, return the first N functions
        def __getitem__(self, key):
            assert key.start is None
            assert key.step is None
            return self._precompute_slice(key.stop)
            
        @overload(int, object) # the second argument is object in order to handle FunctionsList's AdditionalFunctionType
        def __setitem__(self, key, item):
            assert len(self._components) == 1, "Cannot set components, only single functions. Did you mean to call __getitem__ to extract a component and __setitem__ of a single function on that component?"
            assert len(self._components_name) == 1
            self._components[self._components_name[0]][key] = item
        
        @overload(int)
        def _precompute_slice(self, N):
            if N not in self._precomputed_slices:
                assert len(self._components) == 1
                self._precomputed_slices[N] = _BasisFunctionsMatrix.__new__(type(self), self.space)
                self._precomputed_slices[N].__init__(self.space)
                self._precomputed_slices[N].init(self._components_name)
                for component_name in self._components_name:
                    self._precomputed_slices[N]._components[component_name].enrich(self._components[component_name][:N], copy=False)
                    self._precomputed_slices[N]._component_name_to_basis_component_length[component_name] = len(self._precomputed_slices[N]._components[component_name])
            return self._precomputed_slices[N]
            
        @overload(OnlineSizeDict)
        def _precompute_slice(self, N):
            assert set(N.keys()) == set(self._components_name)
            N_key = tuple(N[component_name] for component_name in self._components_name)
            if N_key not in self._precomputed_slices:
                self._precomputed_slices[N_key] = _BasisFunctionsMatrix.__new__(type(self), self.space)
                self._precomputed_slices[N_key].__init__(self.space)
                self._precomputed_slices[N_key].init(self._components_name)
                for component_name in self._components_name:
                    self._precomputed_slices[N_key]._components[component_name].enrich(self._components[component_name][:N[component_name]], copy=False)
                    self._precomputed_slices[N_key]._component_name_to_basis_component_length[component_name] = len(self._precomputed_slices[N_key]._components[component_name])
            return self._precomputed_slices[N_key]
            
        def __iter__(self):
            raise NotImplementedError("BasisFunctionsMatrix.iter() has not been implemented yet")
    return _BasisFunctionsMatrix
