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
## @file solve.py
#  @brief solve function for the solution of a linear system, similar to FEniCS' solve
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from numpy.linalg import solve
from rbnics.backends.abstract import LinearSolver as AbstractLinearSolver
from rbnics.backends.numpy.matrix import Matrix
from rbnics.backends.numpy.vector import Vector
from rbnics.backends.numpy.function import Function
from rbnics.backends.numpy.wrapping import DirichletBC
from rbnics.utils.decorators import BackendFor, DictOfThetaType, Extends, override, ThetaType

@Extends(AbstractLinearSolver)
@BackendFor("numpy", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), ThetaType + DictOfThetaType + (None,)))
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
            self.bcs = DirichletBC(self.lhs, self.rhs, bcs)
            self.bcs.apply_to_vector(self.rhs)
            self.bcs.apply_to_matrix(self.lhs)
                
    @override
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "NumPy linear solver does not accept parameters yet"
                
    @override
    def solve(self):
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        return self.solution
        
