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
from RBniCS.backend.abstract import LinearSolver as AbstractLinearSolver
from RBniCS.backends.fenics.matrix import Matrix_Type
from RBniCS.backends.fenics.vector import Vector_Type
from RBniCS.backends.fenics.function import Function_Type
from RBniCS.utils.decorators import any, BackendFor, Extends, override

@Extends(AbstractLinearSolver)
@BackendFor("FEniCS", inputs=(Matrix_Type, Function_Type, Vector_Type, any(DirichletBC, None)))
class LinearSolver(AbstractLinearSolver):
    @override
    def __init__(lhs, solution, rhs, bcs=None):
        self.lhs = lhs
        self.solution = solution
        self.rhs = rhs
        self.bcs = bcs
        
    @override
    def solve():
        if self.bcs is not None:
            assert isinstance(self.bcs, list)
            for bc in self.bcs:
                bc.apply(self.lhs, self.rhs)
        solve(self.lhs, self.solution.vector(), self.rhs)
        
