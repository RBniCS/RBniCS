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

from rbnics.backends.basic.wrapping.delayed_functions_list import DelayedFunctionsList
from rbnics.backends.basic.wrapping.delayed_linear_solver import DelayedLinearSolver
from rbnics.utils.decorators import overload
from rbnics.utils.io import ComponentNameToBasisComponentIndexDict, OnlineSizeDict

class DelayedBasisFunctionsMatrix(object):
    def __init__(self, space):
        self.space = space
        self._components_name = list()
        self._component_name_to_basis_component_index = ComponentNameToBasisComponentIndexDict()
        self._component_name_to_basis_component_length = OnlineSizeDict()
        self._enrich_memory = dict()
        self._precomputed_slices = dict() # from tuple to FunctionsList
        
    def init(self, components_name):
        assert len(self._components_name) is 0
        self._components_name = components_name
        for (basis_component_index, component_name) in enumerate(components_name):
            self._component_name_to_basis_component_index[component_name] = basis_component_index
            self._component_name_to_basis_component_length[component_name] = 0
            self._enrich_memory[component_name] = DelayedFunctionsList(self.space)
        
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
        # Update component name to basis component length
        self._component_name_to_basis_component_length[component_0] += 1
        # Reset precomputed slices
        self._precomputed_slices.clear()
        # Prepare trivial precomputed slice
        self._prepare_trivial_precomputed_slice()
        
    def _prepare_trivial_precomputed_slice(self):
        if len(self._enrich_memory) == 1:
            assert len(self._components_name) == 1
            component_0 = self._components_name[0]
            precomputed_slice_key = self._component_name_to_basis_component_length[component_0]
        else:
            precomputed_slice_key = list()
            for (component_name, basis_component_index) in sorted(self._component_name_to_basis_component_index.items()):
                precomputed_slice_key.append(self._component_name_to_basis_component_length[component_name])
            precomputed_slice_key = tuple(precomputed_slice_key)
        self._precomputed_slices[precomputed_slice_key] = self
        
    @overload(slice) # e.g. key = :N, return the first N functions
    def __getitem__(self, key):
        assert key.start is None
        assert key.step is None
        return self._precompute_slice(key.stop)
        
    def __len__(self):
        assert len(self._components_name) == 1
        assert len(self._enrich_memory) == 1
        component_0 = self._components_name[0]
        return self._component_name_to_basis_component_length[component_0]
        
    @overload(int)
    def _precompute_slice(self, N):
        if N not in self._precomputed_slices:
            assert len(self._enrich_memory) == 1
            self._precomputed_slices[N] = DelayedBasisFunctionsMatrix(self.space)
            self._precomputed_slices[N].init(self._components_name)
            for (component_name, basis_component_index) in sorted(self._component_name_to_basis_component_index.items()):
                self._precomputed_slices[N]._enrich_memory[component_name].enrich(self._enrich_memory[component_name][:N])
                self._precomputed_slices[N]._component_name_to_basis_component_length[component_name] = len(self._precomputed_slices[N]._enrich_memory[component_name])
        return self._precomputed_slices[N]
        
    @overload(OnlineSizeDict)
    def _precompute_slice(self, N):
        assert set(N.keys()) == set(self._components_name)
        N_key = tuple(N[component_name] for (component_name, basis_component_index) in sorted(self._component_name_to_basis_component_index.items()))
        if N_key not in self._precomputed_slices:
            self._precomputed_slices[N_key] = DelayedBasisFunctionsMatrix(self.space)
            self._precomputed_slices[N_key].init(self._components_name)
            for (component_name, basis_component_index) in sorted(self._component_name_to_basis_component_index.items()):
                self._precomputed_slices[N_key]._enrich_memory[component_name].enrich(self._enrich_memory[component_name][:N[component_name]])
                self._precomputed_slices[N_key]._component_name_to_basis_component_length[component_name] = len(self._precomputed_slices[N_key]._enrich_memory[component_name])
        return self._precomputed_slices[N_key]

    def save(self, directory, filename):
        for (component, memory) in self._enrich_memory.items():
            memory.save(directory, filename + "_" + component)
        
    def load(self, directory, filename):
        for (component, memory) in self._enrich_memory.items():
            memory.load(directory, filename + "_" + component)
            
    def get_problem_name(self):
        problem_name = None
        for (_, memory) in self._enrich_memory.items():
            if problem_name is None:
                problem_name = memory.get_problem_name()
            else:
                assert memory.get_problem_name() == problem_name
        return problem_name
