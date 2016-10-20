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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy.linalg import solve
from RBniCS.backends.abstract import LinearSolver as AbstractLinearSolver
from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.backends.numpy.vector import Vector
from RBniCS.backends.numpy.function import Function
from RBniCS.utils.decorators import BackendFor, DictOfThetaType, Extends, override, ThetaType

@Extends(AbstractLinearSolver)
@BackendFor("NumPy", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), ThetaType + DictOfThetaType + (None,)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.lhs = lhs
        self.solution = solution
        self.rhs = rhs
        self.bcs = bcs
        # We should be solving a square system
        assert self.lhs.M == self.lhs.N
        assert self.lhs.N == self.rhs.N
        # Prepare indices for bcs
        self.bcs_base_index = None
        if self.bcs is not None and isinstance(self.rhs.N, dict):
            # Auxiliary dicts should have been stored in lhs and rhs, and should be consistent
            assert self.lhs._basis_component_index_to_component_name == self.rhs._basis_component_index_to_component_name
            assert self.lhs._component_name_to_basis_component_index == self.rhs._component_name_to_basis_component_index
            assert self.lhs._component_name_to_basis_component_length == self.rhs._component_name_to_basis_component_length
            # Fill in storage
            self.bcs_base_index = dict() # from component name to first index
            current_bcs_base_index = 0
            for (basis_component_index, component_name) in sorted(self.lhs._basis_component_index_to_component_name.iteritems()):
                self.bcs_base_index[component_name] = current_bcs_base_index
                current_bcs_base_index += self.rhs.N[component_name]
                
    @override
    def solve(self):
        if self.bcs is not None:
            assert isinstance(self.bcs, (tuple, dict))
            if isinstance(self.bcs, tuple):
                for (i, bc_i) in enumerate(self.bcs):
                    self.rhs[i] = bc_i
                    self.lhs[i, :] = 0.
                    self.lhs[i, i] = 1.
            elif isinstance(self.bcs, dict):
                assert self.bcs_base_index is not None
                for (component_name, component_bc) in self.bcs.iteritems():
                    for (i, bc_i) in enumerate(component_bc):
                        block_i = self.bcs_base_index[component_name] + i
                        self.rhs[block_i] = bc_i
                        self.lhs[block_i, :] = 0.
                        self.lhs[block_i, block_i] = 1.
            else:
                raise AssertionError("Invalid bc in LinearSolver.solve().")
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        
