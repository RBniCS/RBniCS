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

import types
from numpy import asarray, dot
from numpy.linalg import solve
from scipy.optimize.nonlin import Jacobian, nonlin_solve
from rbnics.backends.abstract import NonlinearSolver as AbstractNonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.numpy.function import Function
from rbnics.backends.numpy.wrapping import DirichletBC
from rbnics.utils.decorators import BackendFor, DictOfThetaType, Extends, override, ThetaType

@Extends(AbstractNonlinearSolver)
@BackendFor("numpy", inputs=(NonlinearProblemWrapper, Function.Type()))
class NonlinearSolver(AbstractNonlinearSolver):
    @override
    def __init__(self, problem_wrapper, solution):
        self.problem = _NonlinearProblem(problem_wrapper.residual_eval, solution, problem_wrapper.bc_eval(), problem_wrapper.jacobian_eval)
        # Additional storage which will be setup by set_parameters
        self._absolute_tolerance = None
        self._line_search = True
        self._maximum_iterations = None
        self._monitor = None
        self._relative_tolerance = None
        self._report = False
        self._solution_tolerance = None
                        
    @override
    def set_parameters(self, parameters):
        for (key, value) in parameters.iteritems():
            if key == "absolute_tolerance":
                self._absolute_tolerance = value
            elif key == "line_search":
                self._line_search = value
            elif key == "maximum_iterations":
                self._maximum_iterations = value
            elif key == "relative_tolerance":
                self._relative_tolerance = value
            elif key == "report":
                self._report = value
            elif key == "solution_tolerance":
                self._solution_tolerance = value
            else:
                raise ValueError("Invalid paramater passed to scipy object.")
                
    @override
    def solve(self):
        residual = self.problem.residual
        initial_guess_vector = asarray(self.problem.solution.vector()).reshape(-1)
        jacobian = _Jacobian(self.problem.jacobian)
        solution_vector = nonlin_solve(
            residual, initial_guess_vector, jacobian=jacobian, verbose=self._report,
            f_tol=self._absolute_tolerance, f_rtol=self._relative_tolerance, x_rtol=self._solution_tolerance, maxiter=self._maximum_iterations,
            line_search=self._line_search, callback=self._monitor
        )
        self.problem.solution.vector()[:] = solution_vector.reshape((-1, 1))
        return self.problem.solution
        
class _NonlinearProblem(object):
    def __init__(self, residual_eval, solution, bcs, jacobian_eval):
        self.residual_eval = residual_eval
        self.solution = solution
        self.jacobian_eval = jacobian_eval
        # We should be solving a square system
        sample_residual = residual_eval(solution)
        sample_jacobian = jacobian_eval(solution)
        assert sample_jacobian.M == sample_jacobian.N
        assert sample_jacobian.N == sample_residual.N
        # Prepare storage for BCs, if necessary
        if bcs is not None:
            self.bcs = DirichletBC(sample_jacobian, sample_residual, bcs)
            # Apply BCs to initial guess
            self.bcs.apply_to_vector(self.solution.vector())
        else:
            self.bcs = None
        
    def residual(self, solution):
        # Convert to a matrix with one column, rather than an array
        self.solution.vector()[:] = solution.reshape((-1, 1))
        # Compute residual
        residual_vector = self.residual_eval(self.solution)
        # Apply BCs, if necessary
        if self.bcs is not None:
            self.bcs.homogeneous_apply_to_vector(residual_vector)
        # Convert to an array, rather than a matrix with one column, and return
        return asarray(residual_vector).reshape(-1)
        
    def jacobian(self, solution):
        # Convert to a matrix with one column, rather than an array
        self.solution.vector()[:] = solution.reshape((-1, 1))
        # Compute jacobian
        jacobian_matrix = self.jacobian_eval(self.solution)
        # Apply BCs, if necessary
        if self.bcs is not None:
            self.bcs.apply_to_matrix(jacobian_matrix)
        # Return
        return jacobian_matrix
        
# Adapted from scipy/optimize/nonlin.py, asjacobian method
class _Jacobian(Jacobian):
    def __init__(self, jacobian_eval):
        self.jacobian_eval = jacobian_eval
    
    def setup(self, x, F, func):
        Jacobian.setup(self, x, F, func)
        self.x = x
    
    def update(self, x, F):
        self.x = x

    def solve(self, v, tol=0):
        J = self.jacobian_eval(self.x)
        return solve(J, v)
        
    def matvec(self, v):
        J = self.jacobian_eval(self.x)
        return dot(J, v)

    def rsolve(self, v, tol=0):
        J = self.jacobian_eval(self.x)
        return solve(J.T, v)

    def rmatvec(self, v):
        J = self.jacobian_eval(self.x)
        return dot(J.T, v)
        
