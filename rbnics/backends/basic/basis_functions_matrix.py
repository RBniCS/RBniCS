# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from rbnics.backends.abstract import BasisFunctionsMatrix as AbstractBasisFunctionsMatrix
from rbnics.utils.cache import Cache
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
            self._components = dict()  # of FunctionsList
            self._precomputed_sub_components = Cache()  # from tuple to FunctionsList
            self._precomputed_slices = Cache()  # from tuple to FunctionsList
            self._components_name = list()  # filled in by init
            self._component_name_to_basis_component_index = ComponentNameToBasisComponentIndexDict()  # filled by init
            self._component_name_to_basis_component_length = OnlineSizeDict()

        def init(self, components_name):

            # Helper function to patch FunctionsList.enrich() to update internal attributes
            def patch_functions_list_enrich(component_name, functions_list):
                original_functions_list_enrich = functions_list.enrich

                def patched_functions_list_enrich(self_, functions, component=None, weights=None, copy=True):
                    # Append to storage
                    original_functions_list_enrich(functions, component, weights, copy)
                    # Update component name to basis component length
                    if component is not None:
                        if isinstance(component, dict):
                            assert len(component) == 1
                            for (_, component_to) in component.items():
                                break
                            assert component_name == component_to
                        else:
                            assert component_name == component
                    self._update_component_name_to_basis_component_length(component_name)
                    # Reset precomputed sub components
                    self._precomputed_sub_components.clear()
                    # Prepare trivial precomputed sub components
                    self._prepare_trivial_precomputed_sub_components()
                    # Reset precomputed slices
                    self._precomputed_slices.clear()
                    # Prepare trivial precomputed slice
                    self._prepare_trivial_precomputed_slice()

                functions_list.enrich_patch = PatchInstanceMethod(functions_list, "enrich",
                                                                  patched_functions_list_enrich)
                functions_list.enrich_patch.patch()

            if self._components_name != components_name:  # Do nothing if it was already initialized with the same dicts
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
                # Reset precomputed sub components
                self._precomputed_sub_components.clear()
                # Reset precomputed slices
                self._precomputed_slices.clear()
                # Patch FunctionsList.enrich() to update internal attributes
                for component_name in components_name:
                    patch_functions_list_enrich(component_name, self._components[component_name])

        def enrich(self, functions, component=None, weights=None, copy=True):
            assert copy is True
            # Append to storage
            self._enrich(functions, component, weights, copy)

        # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        @overload(object, None, (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            assert len(self._components) == 1
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            self._components[component_0].enrich(functions, None, weights, copy)

        # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        @overload(object, str, (None, list_of(Number)), bool)
        def _enrich(self, functions, component, weights, copy):
            assert component in self._components
            self._components[component].enrich(functions, component, weights, copy)

        # the first argument is object in order to handle FunctionsList's AdditionalFunctionType
        @overload(object, dict_of(str, str), (None, list_of(Number)), bool)
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

        def _prepare_trivial_precomputed_sub_components(self):
            self._precomputed_sub_components[tuple(self._components_name)] = self

        def _prepare_trivial_precomputed_slice(self):
            if len(self._components) == 1:
                assert len(self._components_name) == 1
                component_0 = self._components_name[0]
                precomputed_slice_key_start = 0
                precomputed_slice_key_stop = self._component_name_to_basis_component_length[component_0]
            else:
                precomputed_slice_key_start = list()
                precomputed_slice_key_stop = list()
                for component_name in self._components_name:
                    precomputed_slice_key_start.append(0)
                    precomputed_slice_key_stop.append(self._component_name_to_basis_component_length[component_name])
                precomputed_slice_key_start = tuple(precomputed_slice_key_start)
                precomputed_slice_key_stop = tuple(precomputed_slice_key_stop)
            self._precomputed_slices[precomputed_slice_key_start, precomputed_slice_key_stop] = self

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
            for (component_name, functions_list) in self._components.items():
                functions_list.save(directory, filename_and_component(component_name))

        def load(self, directory, filename):
            return_value = True
            assert len(self._components) > 0
            if len(self._components) > 1:
                def filename_and_component(component_name):
                    return filename + "_" + component_name
            else:
                def filename_and_component(component_name):
                    return filename
            for (component_name, functions_list) in self._components.items():
                # Skip updating internal attributes while reading in basis functions, we will do that
                # only once at the end
                assert hasattr(functions_list, "enrich_patch")
                functions_list.enrich_patch.unpatch()
                # Load each component
                return_value_component = functions_list.load(directory, filename_and_component(component_name))
                return_value = return_value and return_value_component
                # Populate component length
                self._update_component_name_to_basis_component_length(component_name)
                # Restore patched enrich method
                functions_list.enrich_patch.patch()
            # Reset precomputed sub components
            self._precomputed_sub_components.clear()
            # Prepare trivial precomputed sub components
            self._prepare_trivial_precomputed_sub_components()
            # Reset precomputed slices
            self._precomputed_slices.clear()
            # Prepare trivial precomputed slice
            self._prepare_trivial_precomputed_slice()
            # Return
            return return_value

        @overload(online_backend.OnlineMatrix.Type(), )
        def __mul__(self, other):

            def BasisFunctionsMatrixWithInit(space):
                output = _BasisFunctionsMatrix.__new__(type(self), space)
                output.__init__(space)
                output.init(self._components_name)
                return output

            if isinstance(other.M, dict):
                assert set(other.M.keys()) == set(self._components_name)
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
            # return all basis functions for each component, then the user may use __getitem__ of FunctionsList
            # to extract a single basis function
            return self._components[key]

        @overload(list_of(str))
        def __getitem__(self, key):
            return self._precompute_sub_components(key)

        @overload(slice)  # e.g. key = :N, return the first N functions
        def __getitem__(self, key):
            assert key.step is None
            return self._precompute_slice(key.start, key.stop)

        # the second argument is object in order to handle FunctionsList's AdditionalFunctionType
        @overload(int, object)
        def __setitem__(self, key, item):
            assert len(self._components) == 1, (
                "Cannot set components, only single functions. "
                "Did you mean to call __getitem__ to extract a component and __setitem__ of a single function"
                "on that component?")
            assert len(self._components_name) == 1
            self._components[self._components_name[0]][key] = item

        @overload(None, int)
        def _precompute_slice(self, _, N_stop):
            return self._precompute_slice(0, N_stop)

        @overload(int, None)
        def _precompute_slice(self, N_start, _):
            return self._precompute_slice(N_start, len(self))

        @overload(int, int)
        def _precompute_slice(self, N_start, N_stop):
            if (N_start, N_stop) not in self._precomputed_slices:
                assert len(self._components) == 1
                output = _BasisFunctionsMatrix.__new__(type(self), self.space)
                output.__init__(self.space)
                output.init(self._components_name)
                for component_name in self._components_name:
                    output._components[component_name].enrich(self._components[component_name][N_start:N_stop],
                                                              copy=False)
                self._precomputed_slices[N_start, N_stop] = output
            return self._precomputed_slices[N_start, N_stop]

        @overload(None, OnlineSizeDict)
        def _precompute_slice(self, _, N_stop):
            N_start = OnlineSizeDict()
            for component_name in self._components_name:
                N_start[component_name] = 0
            return self._precompute_slice(N_start, N_stop)

        @overload(OnlineSizeDict, None)
        def _precompute_slice(self, N_start, _):
            N_stop = OnlineSizeDict()
            for component_name in self._components_name:
                N_stop[component_name] = self._component_name_to_basis_component_length[component_name]
            return self._precompute_slice(N_start, len(self))

        @overload(OnlineSizeDict, OnlineSizeDict)
        def _precompute_slice(self, N_start, N_stop):
            assert set(N_start.keys()) == set(self._components_name)
            assert set(N_stop.keys()) == set(self._components_name)
            N_start_key = tuple(N_start[component_name] for component_name in self._components_name)
            N_stop_key = tuple(N_stop[component_name] for component_name in self._components_name)
            if (N_start_key, N_stop_key) not in self._precomputed_slices:
                output = _BasisFunctionsMatrix.__new__(type(self), self.space)
                output.__init__(self.space)
                output.init(self._components_name)
                for component_name in self._components_name:
                    output._components[component_name].enrich(self._components[component_name][
                        N_start[component_name]:N_stop[component_name]], copy=False)
                self._precomputed_slices[N_start_key, N_stop_key] = output
            return self._precomputed_slices[N_start_key, N_stop_key]

        def _precompute_sub_components(self, sub_components):
            sub_components_key = tuple(sub_components)
            if sub_components_key not in self._precomputed_sub_components:
                assert set(sub_components).issubset(self._components_name)
                output = _BasisFunctionsMatrix.__new__(type(self), self.space, sub_components)
                output.__init__(self.space, sub_components)
                output.init(sub_components)
                for component_name in sub_components:
                    output._components[component_name].enrich(self._components[component_name],
                                                              component=component_name, copy=True)
                self._precomputed_sub_components[sub_components_key] = output
            return self._precomputed_sub_components[sub_components_key]

        def __iter__(self):
            assert len(self._components) == 1
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            return self._components[component_0].__iter__()

    return _BasisFunctionsMatrix
