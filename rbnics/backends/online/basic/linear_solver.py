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

from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver
from rbnics.backends.online.basic.wrapping import DirichletBC
from rbnics.utils.decorators import Extends, override

@Extends(AbstractLinearSolver)
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.lhs = lhs
        self.solution = solution
        self.rhs = rhs
        # We should be solving a square system
        assert self.lhs.M == self.lhs.N
        assert self.lhs.N == self.rhs.N
        # Apply BCs, if necessary
        if bcs is not None:
            assert isinstance(bcs, (tuple, dict))
            if isinstance(bcs, tuple):
                self.bcs = DirichletBC(bcs)
            elif isinstance(bcs, dict):
                # Auxiliary dicts should have been stored in lhs and rhs, and should be consistent
                assert self.rhs._basis_component_index_to_component_name == self.lhs._basis_component_index_to_component_name[1]
                assert self.rhs._component_name_to_basis_component_index == self.lhs._component_name_to_basis_component_index[1]
                assert self.rhs._component_name_to_basis_component_length == self.lhs._component_name_to_basis_component_length[1]
                # Provide auxiliary dicts to DirichletBC constructor
                self.bcs = DirichletBC(bcs, self.rhs._basis_component_index_to_component_name, self.rhs.N)
            else:
                raise AssertionError("Invalid bc in LinearSolver.__init__().")
            self.bcs.apply_to_vector(self.rhs)
            self.bcs.apply_to_matrix(self.lhs)
