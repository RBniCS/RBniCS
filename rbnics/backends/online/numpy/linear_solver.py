# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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


@BackendFor("numpy", inputs=((Matrix.Type(), DelayedTransposeWithArithmetic, LinearProblemWrapper),
                             Function.Type(),
                             (Vector.Type(), DelayedTransposeWithArithmetic, None),
                             ThetaType + DictOfThetaType + (None,)))
class LinearSolver(LinearSolver_Base):
    def set_parameters(self, parameters):
        assert len(parameters) == 0, "NumPy linear solver does not accept parameters yet"

    def solve(self):
        solution = solve(self.lhs, self.rhs)
        self.solution.vector()[:] = solution
        if self.monitor is not None:
            self.monitor(self.solution)
