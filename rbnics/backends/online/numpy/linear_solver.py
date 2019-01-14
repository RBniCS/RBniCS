# Copyright (C) 2015-2019 by the RBniCS authors
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
from rbnics.backends.abstract import LinearProblemWrapper
from rbnics.backends.online.basic import LinearSolver as BasicLinearSolver
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.transpose import DelayedTransposeWithArithmetic
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import BackendFor, DictOfThetaType, ModuleWrapper, ThetaType

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(DelayedTransposeWithArithmetic=DelayedTransposeWithArithmetic)
LinearSolver_Base = BasicLinearSolver(backend, wrapping)

@BackendFor("numpy", inputs=((Matrix.Type(), DelayedTransposeWithArithmetic, LinearProblemWrapper), Function.Type(), (Vector.Type(), DelayedTransposeWithArithmetic, None), ThetaType + DictOfThetaType + (None,)))
class LinearSolver(LinearSolver_Base):
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "NumPy linear solver does not accept parameters yet"
        
    def solve(self):
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        if self.monitor is not None:
            self.monitor(self.solution)
