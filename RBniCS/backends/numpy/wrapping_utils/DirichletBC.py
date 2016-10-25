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

class DirichletBC(object):
    def __init__(self, lhs, rhs, bcs):
        self.bcs = bcs
        assert isinstance(self.bcs, (tuple, dict))
        if isinstance(self.bcs, tuple):
            # No additional storage needed
            self.bcs_base_index = None
        elif isinstance(self.bcs, dict):
            # Auxiliary dicts should have been stored in lhs and rhs, and should be consistent
            assert rhs._basis_component_index_to_component_name == lhs._basis_component_index_to_component_name
            assert rhs._component_name_to_basis_component_index == lhs._component_name_to_basis_component_index
            assert rhs._component_name_to_basis_component_length == lhs._component_name_to_basis_component_length
            assert rhs.N == lhs.N
            # Fill in storage
            bcs_base_index = dict() # from component name to first index
            current_bcs_base_index = 0
            for (basis_component_index, component_name) in sorted(lhs._basis_component_index_to_component_name.iteritems()):
                bcs_base_index[component_name] = current_bcs_base_index
                current_bcs_base_index += rhs.N[component_name]
            self.bcs_base_index = bcs_base_index
        else:
            raise AssertionError("Invalid bc in DirichletBC.__init__().")
            
    def apply_to_vector(self, vector):
        if isinstance(self.bcs, tuple):
            # Apply BCs to the increment
            for (i, bc_i) in enumerate(self.bcs):
                vector[i] = bc_i
        elif isinstance(self.bcs, dict):
            # Apply BCs to the increment
            for (component_name, component_bc) in self.bcs.iteritems():
                for (i, bc_i) in enumerate(component_bc):
                    block_i = bcs_base_index[component_name] + i
                    vector[block_i] = bc_i
        else:
            raise AssertionError("Invalid bc in DirichletBC.apply_to_vector().")
        
    def homogeneous_apply_to_vector(self, vector):
        if isinstance(self.bcs, tuple):
            # Apply BCs to the increment
            for (i, _) in enumerate(self.bcs):
                vector[i] = 0.
        elif isinstance(self.bcs, dict):
            # Apply BCs to the increment
            for (component_name, component_bc) in self.bcs.iteritems():
                for (i, _) in enumerate(component_bc):
                    block_i = bcs_base_index[component_name] + i
                    vector[block_i] = 0.
        else:
            raise AssertionError("Invalid bc in DirichletBC.homogeneous_apply_to_vector().")
        
    def apply_to_matrix(self, matrix):
        if isinstance(self.bcs, tuple):
            # Apply BCs
            for (i, _) in enumerate(self.bcs):
                matrix[i, :] = 0.
                matrix[i, i] = 1.
        elif isinstance(self.bcs, dict):
            # Apply BCs
            for (component_name, component_bc) in self.bcs.iteritems():
                for (i, _) in enumerate(component_bc):
                    block_i = bcs_base_index[component_name] + i
                    matrix[block_i, :] = 0.
                    matrix[block_i, block_i] = 1.
        else:
            raise AssertionError("Invalid bc in DirichletBC.apply_to_matrix().")
        

    
