# Copyright (C) 2015-2019 by the RBniCS authors
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

from rbnics.backends.basic.wrapping.delayed_functions_list import DelayedFunctionsList
from rbnics.backends.basic.wrapping.delayed_linear_solver import DelayedLinearSolver
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import overload
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict
from rbnics.utils.test import PatchInstanceMethod

class DelayedBasisFunctionsMatrix(object):
    def __init__(self, space):
        self.space = space
        self._components_name = list()
        self._component_name_to_basis_component_index = ComponentNameToBasisComponentIndexDict()
        self._component_name_to_basis_component_length = OnlineSizeDict()
        self._enrich_memory = Cache()
        self._precomputed_slices = Cache() # from tuple to FunctionsList
        
    def init(self, components_name):
        # Patch DelayedFunctionsList.enrich() to update internal attributes
        def patch_delayed_functions_list_enrich(component_name, memory):
            original_delayed_functions_list_enrich = memory.enrich
            def patched_delayed_functions_list_enrich(self_, functions, component=None, weights=None, copy=True):
                # Append to storage
                original_delayed_functions_list_enrich(functions, component, weights, copy)
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
                # Reset precomputed slices
                self._precomputed_slices.clear()
                # Prepare trivial precomputed slice
                self._prepare_trivial_precomputed_slice()
            memory.enrich_patch = PatchInstanceMethod(memory, "enrich", patched_delayed_functions_list_enrich)
            memory.enrich_patch.patch()
            
        assert len(self._components_name) == 0
        self._components_name = components_name
        for (basis_component_index, component_name) in enumerate(components_name):
            self._component_name_to_basis_component_index[component_name] = basis_component_index
            self._component_name_to_basis_component_length[component_name] = 0
            self._enrich_memory[component_name] = DelayedFunctionsList(self.space)
            patch_delayed_functions_list_enrich(component_name, self._enrich_memory[component_name])
        
    def enrich(self, function, component=None, weight=None, copy=True):
        assert isinstance(function, DelayedLinearSolver)
        assert component is None
        assert weight is None
        assert copy is True
        assert len(self._components_name) == 1
        assert len(self._enrich_memory) == 1
        component_0 = self._components_name[0]
        # Append to storage
        self._enrich_memory[component_0].enrich(function, component, weight, copy)
        
    @overload(None)
    def _update_component_name_to_basis_component_length(self, component):
        assert len(self._components) == 1
        assert len(self._components_name) == 1
        component_0 = self._components_name[0]
        self._component_name_to_basis_component_length[component_0] = len(self._enrich_memory[component_0])
        
    @overload(str)
    def _update_component_name_to_basis_component_length(self, component):
        self._component_name_to_basis_component_length[component] = len(self._enrich_memory[component])
        
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
        
    @overload(slice) # e.g. key = :N, return the first N functions
    def __getitem__(self, key):
        assert key.step is None
        return self._precompute_slice(key.start, key.stop)
        
    @overload(str)
    def __getitem__(self, key):
        return self._enrich_memory[key]
        
    def __len__(self):
        assert len(self._components_name) == 1
        assert len(self._enrich_memory) == 1
        component_0 = self._components_name[0]
        return self._component_name_to_basis_component_length[component_0]
        
    @overload(None, int)
    def _precompute_slice(self, _, N_stop):
        return self._precompute_slice(0, N_stop)
        
    @overload(int, None)
    def _precompute_slice(self, N_start, _):
        return self._precompute_slice(N_start, len(self))
        
    @overload(int, int)
    def _precompute_slice(self, N_start, N_stop):
        if (N_start, N_stop) not in self._precomputed_slices:
            assert len(self._enrich_memory) == 1
            output = DelayedBasisFunctionsMatrix(self.space)
            output.init(self._components_name)
            for component_name in self._components_name:
                output._enrich_memory[component_name].enrich(self._enrich_memory[component_name][N_start:N_stop])
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
            output = DelayedBasisFunctionsMatrix(self.space)
            output.init(self._components_name)
            for component_name in self._components_name:
                output._enrich_memory[component_name].enrich(self._enrich_memory[component_name][N_start[component_name]:N_stop[component_name]])
            self._precomputed_slices[N_start_key, N_stop_key] = output
        return self._precomputed_slices[N_start_key, N_stop_key]

    def save(self, directory, filename):
        for (component, memory) in self._enrich_memory.items():
            memory.save(directory, filename + "_" + component)
        
    def load(self, directory, filename):
        return_value = True
        for (component, memory) in self._enrich_memory.items():
            # Skip updating internal attributes while reading in basis functions, we will do that
            # only once at the end
            assert hasattr(memory, "enrich_patch")
            memory.enrich_patch.unpatch()
            # Load each component
            return_value_component = memory.load(directory, filename + "_" + component)
            return_value = return_value and return_value_component
            # Populate component length
            self._update_component_name_to_basis_component_length(component)
            # Restore patched enrich method
            memory.enrich_patch.patch()
        # Reset precomputed slices
        self._precomputed_slices.clear()
        # Prepare trivial precomputed slice
        self._prepare_trivial_precomputed_slice()
        return return_value
        
    def get_problem_name(self):
        problem_name = None
        for (_, memory) in self._enrich_memory.items():
            if problem_name is None:
                problem_name = memory.get_problem_name()
            else:
                assert memory.get_problem_name() == problem_name
        return problem_name
