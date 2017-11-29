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

from numpy.linalg import solve
from rbnics.backends.online.basic import LinearSolver as BasicLinearSolver
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.function import Function
from rbnics.utils.decorators import BackendFor, DictOfThetaType, ThetaType

LinearSolver_Base = BasicLinearSolver

@BackendFor("numpy", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), ThetaType + DictOfThetaType + (None,)))
class LinearSolver(LinearSolver_Base):
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "NumPy linear solver does not accept parameters yet"
        
    def solve(self):
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        # Preserve auxiliary attributes related to basis functions matrix
        assert hasattr(self.lhs, "_basis_component_index_to_component_name") == hasattr(self.lhs, "_component_name_to_basis_component_index")
        assert hasattr(self.lhs, "_basis_component_index_to_component_name") == hasattr(self.lhs, "_component_name_to_basis_component_length")
        assert hasattr(self.rhs, "_basis_component_index_to_component_name") == hasattr(self.rhs, "_component_name_to_basis_component_index")
        assert hasattr(self.rhs, "_basis_component_index_to_component_name") == hasattr(self.rhs, "_component_name_to_basis_component_length")
        assert hasattr(self.lhs, "_basis_component_index_to_component_name") == hasattr(self.rhs, "_basis_component_index_to_component_name")
        if hasattr(self.rhs, "_basis_component_index_to_component_name"):
            self.solution.vector()._basis_component_index_to_component_name = self.lhs._basis_component_index_to_component_name[0]
            self.solution.vector()._component_name_to_basis_component_index = self.lhs._component_name_to_basis_component_index[0]
            self.solution.vector()._component_name_to_basis_component_length = self.lhs._component_name_to_basis_component_length[0]
        # Return
        return self.solution
