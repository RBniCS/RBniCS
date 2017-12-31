# Copyright (C) 2015-2018 by the RBniCS authors
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

from numpy import dot
from numpy.linalg import solve
from scipy.optimize.nonlin import Jacobian, nonlin_solve
from rbnics.backends.abstract import NonlinearSolver as AbstractNonlinearSolver, NonlinearProblemWrapper
from rbnics.backends.online.basic.nonlinear_solver import _NonlinearProblem as _BasicNonlinearProblem
from rbnics.backends.online.numpy.function import Function
from rbnics.utils.decorators import BackendFor

@BackendFor("numpy", inputs=(NonlinearProblemWrapper, Function.Type()))
class NonlinearSolver(AbstractNonlinearSolver):
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
                        
    def set_parameters(self, parameters):
        for (key, value) in parameters.items():
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
                
    def solve(self):
        residual = self.problem.residual
        initial_guess_vector = self.problem.solution.vector()
        jacobian = _Jacobian(self.problem.jacobian)
        solution_vector = nonlin_solve(
            residual, initial_guess_vector, jacobian=jacobian, verbose=self._report,
            f_tol=self._absolute_tolerance, f_rtol=self._relative_tolerance, x_rtol=self._solution_tolerance, maxiter=self._maximum_iterations,
            line_search=self._line_search, callback=self._monitor
        )
        self.problem.solution.vector()[:] = solution_vector
        return self.problem.solution
        
class _NonlinearProblem(_BasicNonlinearProblem):
    def residual(self, solution):
        # Store solution
        self.solution.vector()[:] = solution
        # Compute residual
        residual_vector = self.residual_eval(self.solution)
        # Apply BCs, if necessary
        if self.bcs is not None:
            self.bcs.apply_to_vector(residual_vector, self.solution.vector())
        # Convert to an array, rather than a matrix with one column, and return
        return residual_vector
        
    def jacobian(self, solution):
        # Store solution
        self.solution.vector()[:] = solution
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
