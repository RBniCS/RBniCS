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

from dolfin import DirichletBC, solve
from RBniCS.backends.abstract import LinearSolver as AbstractLinearSolver
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.utils.decorators import BackendFor, Extends, list_of, override

@Extends(AbstractLinearSolver)
@BackendFor("FEniCS", inputs=(Matrix.Type(), Function.Type(), Vector.Type(), (list_of(DirichletBC), None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(self, lhs, solution, rhs, bcs=None):
        self.solution = solution
        if bcs is not None:
            # Create a copy of lhs and rhs, in order not to 
            # change the original references when applying bcs
            self.lhs = lhs.copy()
            self.rhs = rhs.copy()
            self.bcs = bcs
        else:
            self.lhs = lhs
            self.rhs = rhs
            self.bcs = bcs
        
    @override
    def solve(self):
        if self.bcs is not None:
            assert isinstance(self.bcs, list)
            for bc in self.bcs:
                assert isinstance(bc, DirichletBC)
                bc.apply(self.lhs, self.rhs)
        solve(self.lhs, self.solution.vector(), self.rhs)
        
