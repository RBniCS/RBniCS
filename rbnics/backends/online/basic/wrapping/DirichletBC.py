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

class DirichletBC(object):
    def __init__(self, bcs, basis_component_index_to_component_name=None, N=None):
        if bcs is not None:
            self.bcs = bcs
            assert isinstance(self.bcs, (tuple, dict))
            if isinstance(self.bcs, tuple):
                assert basis_component_index_to_component_name is None
                assert N is None
                # No additional storage needed
                self.bcs_base_index = None
            elif isinstance(self.bcs, dict):
                assert basis_component_index_to_component_name is not None
                assert N is not None
                assert isinstance(basis_component_index_to_component_name, dict)
                assert isinstance(N, dict)
                # Fill in storage
                bcs_base_index = dict() # from component name to first index
                current_bcs_base_index = 0
                for (basis_component_index, component_name) in sorted(basis_component_index_to_component_name.items()):
                    bcs_base_index[component_name] = current_bcs_base_index
                    current_bcs_base_index += N[component_name]
                self.bcs_base_index = bcs_base_index
            else:
                raise AssertionError("Invalid bc in DirichletBC.__init__().")
        else:
            self.bcs = None
            
    def apply_to_vector(self, vector, solution=None):
        if self.bcs is not None:
            if solution is None:
                if isinstance(self.bcs, tuple):
                    for (i, bc_i) in enumerate(self.bcs):
                        vector[i] = bc_i
                elif isinstance(self.bcs, dict):
                    for (component_name, component_bc) in self.bcs.items():
                        for (i, bc_i) in enumerate(component_bc):
                            block_i = self.bcs_base_index[component_name] + i
                            vector[block_i] = bc_i
                else:
                    raise AssertionError("Invalid bc in DirichletBC.apply_to_vector().")
            else:
                if isinstance(self.bcs, tuple):
                    for (i, bc_i) in enumerate(self.bcs):
                        vector[i] = float(solution[i]) - bc_i
                elif isinstance(self.bcs, dict):
                    for (component_name, component_bc) in self.bcs.items():
                        for (i, bc_i) in enumerate(component_bc):
                            block_i = self.bcs_base_index[component_name] + i
                            vector[block_i] = float(solution[block_i]) - bc_i
                else:
                    raise AssertionError("Invalid bc in DirichletBC.apply_to_vector().")
        
    def homogeneous_apply_to_vector(self, vector):
        if self.bcs is not None:
            if isinstance(self.bcs, tuple):
                for (i, _) in enumerate(self.bcs):
                    vector[i] = 0.
            elif isinstance(self.bcs, dict):
                for (component_name, component_bc) in self.bcs.items():
                    for (i, _) in enumerate(component_bc):
                        block_i = self.bcs_base_index[component_name] + i
                        vector[block_i] = 0.
            else:
                raise AssertionError("Invalid bc in DirichletBC.homogeneous_apply_to_vector().")
        
    def apply_to_matrix(self, matrix):
        if self.bcs is not None:
            if isinstance(self.bcs, tuple):
                for (i, _) in enumerate(self.bcs):
                    matrix[i, :] = 0.
                    matrix[i, i] = 1.
            elif isinstance(self.bcs, dict):
                for (component_name, component_bc) in self.bcs.items():
                    for (i, _) in enumerate(component_bc):
                        block_i = self.bcs_base_index[component_name] + i
                        matrix[block_i, :] = 0.
                        matrix[block_i, block_i] = 1.
            else:
                raise AssertionError("Invalid bc in DirichletBC.apply_to_matrix().")    
