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

from rbnics.utils.decorators import dict_of, DictOfThetaType, overload, ThetaType
from rbnics.utils.io import OnlineSizeDict

# Implementation for empty bcs
@overload(None, None, None)
def DirichletBC(bcs, basis_component_index_to_component_name=None, N=None):
    return _DirichletBC_Empty(bcs, basis_component_index_to_component_name, N)

class _DirichletBC_Empty(object):
    def __init__(self, bcs, basis_component_index_to_component_name=None, N=None):
        pass
        
    def apply_to_vector(self, vector, solution=None):
        pass
        
    def homogeneous_apply_to_vector(self, vector):
        pass
        
    def apply_to_matrix(self, matrix):
        pass

# Implementation for ThetaType
@overload(ThetaType, None, None)
def DirichletBC(bcs, basis_component_index_to_component_name=None, N=None):
    return _DirichletBC_ThetaType(bcs, basis_component_index_to_component_name, N)

class _DirichletBC_ThetaType(object):
    def __init__(self, bcs, basis_component_index_to_component_name=None, N=None):
        self.bcs = bcs
        
    def apply_to_vector(self, vector, solution=None):
        if solution is None:
            for (i, bc_i) in enumerate(self.bcs):
                vector[i] = bc_i
        else:
            for (i, bc_i) in enumerate(self.bcs):
                vector[i] = float(solution[i]) - bc_i
        
    def homogeneous_apply_to_vector(self, vector):
        for (i, _) in enumerate(self.bcs):
            vector[i] = 0.
        
    def apply_to_matrix(self, matrix):
        for (i, _) in enumerate(self.bcs):
            matrix[i, :] = 0.
            matrix[i, i] = 1.

# Implementation for DictOfThetaType
@overload(DictOfThetaType, dict_of(int, str), (dict_of(str, int), OnlineSizeDict))
def DirichletBC(bcs, basis_component_index_to_component_name=None, N=None):
    return _DirichletBC_DictOfThetaType(bcs, basis_component_index_to_component_name, N)

class _DirichletBC_DictOfThetaType(object):
    def __init__(self, bcs, basis_component_index_to_component_name=None, N=None):
        self.bcs = bcs
        bcs_base_index = dict() # from component name to first index
        current_bcs_base_index = 0
        for (basis_component_index, component_name) in sorted(basis_component_index_to_component_name.items()):
            bcs_base_index[component_name] = current_bcs_base_index
            current_bcs_base_index += N[component_name]
        self.bcs_base_index = bcs_base_index
        
    def apply_to_vector(self, vector, solution=None):
        if solution is None:
            for (component_name, component_bc) in self.bcs.items():
                for (i, bc_i) in enumerate(component_bc):
                    block_i = self.bcs_base_index[component_name] + i
                    vector[block_i] = bc_i
        else:
            for (component_name, component_bc) in self.bcs.items():
                for (i, bc_i) in enumerate(component_bc):
                    block_i = self.bcs_base_index[component_name] + i
                    vector[block_i] = float(solution[block_i]) - bc_i
        
    def homogeneous_apply_to_vector(self, vector):
        for (component_name, component_bc) in self.bcs.items():
            for (i, _) in enumerate(component_bc):
                block_i = self.bcs_base_index[component_name] + i
                vector[block_i] = 0.
        
    def apply_to_matrix(self, matrix):
        for (component_name, component_bc) in self.bcs.items():
            for (i, _) in enumerate(component_bc):
                block_i = self.bcs_base_index[component_name] + i
                matrix[block_i, :] = 0.
                matrix[block_i, block_i] = 1.
