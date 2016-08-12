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

from RBniCS.backends.abstract import LinearSolver as AbstractLinearSolver
from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.backends.numpy.vector import Vector
from RBniCS.backends.numpy.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, override, ThetaType

@Extends(AbstractLinearSolver)
@BackendFor("NumPy", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), (ThetaType, None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.lhs = lhs
        self.solution = solution
        self.rhs = rhs
        self.bcs = bcs
        
    @override
    def solve(self):
        if self.bcs is not None:
            assert isinstance(self.bcs, tuple)
            for (i, bc_i) in enumerate(self.bcs):
                self.rhs[i] = bc_i
                self.lhs[i, :] = 0.
                self.lhs[i, i] = 1.
        from numpy.linalg import solve
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        
