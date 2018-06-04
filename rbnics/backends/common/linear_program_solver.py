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

linear_programming_backends = {
    "cvxopt": None,
    "scipy": None
}

try:
    import cvxopt
except ImportError:
    linear_programming_backends["cvxopt"] = False
else:
    linear_programming_backends["cvxopt"] = True

try:
    from scipy.optimize import linprog
except ImportError:
    linear_programming_backends["scipy"] = False
else:
    linear_programming_backends["scipy"] = True
    
from rbnics.backends.abstract import LinearProgramSolver as AbstractLinearProgramSolver
from rbnics.utils.decorators import BackendFor, list_of, tuple_of

# Helper classes for linear pogram
from numpy import eye, hstack, matrix as numpy_matrix, ndarray as numpy_vector, vstack, zeros
def Matrix(m, n):
    return numpy_matrix(zeros((m, n)))
def Vector(n):
    return zeros((n, ))
class Error(RuntimeError):
    pass

if linear_programming_backends["cvxopt"]:
    class CVXOPTLinearProgramSolver(AbstractLinearProgramSolver):
        def __init__(self, cost, inequality_constraints_matrix, inequality_constraints_vector, bounds):
            self.Q = len(cost)
            # Store cost
            self.cost = cvxopt.matrix(cost)
            # Store inequality constraints matrix, also including a 2*Q x 2*Q submatrix for bound constraints
            self.inequality_constraints_matrix = cvxopt.matrix(vstack((- inequality_constraints_matrix, - eye(self.Q), eye(self.Q))))
            # Store inequality constraints vector, also including 2*Q rows for bound constraints
            assert len(bounds) == self.Q
            bounds_lower = zeros(self.Q)
            bounds_upper = zeros(self.Q)
            for (q, bounds_q) in enumerate(bounds):
                assert bounds_q[0] <= bounds_q[1]
                bounds_lower[q] = bounds_q[0]
                bounds_upper[q] = bounds_q[1]
            self.inequality_constraints_vector = cvxopt.matrix(hstack((- inequality_constraints_vector, bounds_lower, bounds_upper)))
            
        def solve(self):
            result = cvxopt.solvers.lp(self.cost, self.inequality_constraints_matrix, self.inequality_constraints_vector, solver="glpk", options={"glpk": {"msg_lev": "GLP_MSG_OFF"}})
            if result["status"] != "optimal":
                raise Error("Linear program solver reports convergence failure with reason", result["status"])
            else:
                return result["primal objective"]
    
if linear_programming_backends["scipy"]:
    class SciPyLinearProgramSolver(AbstractLinearProgramSolver):
        def __init__(self, cost, inequality_constraints_matrix, inequality_constraints_vector, bounds):
            self.cost = cost
            self.inequality_constraints_matrix = - inequality_constraints_matrix
            self.inequality_constraints_vector = - inequality_constraints_vector
            self.bounds = bounds
            
        def solve(self):
            result = linprog(self.cost, self.inequality_constraints_matrix, self.inequality_constraints_vector, bounds=self.bounds)
            if not result.success:
                raise Error("Linear program solver reports convergence failure with reason", result.status)
            else:
                return result.fun
            
from numbers import Number
if linear_programming_backends["cvxopt"]:
    @BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(Number))))
    class LinearProgramSolver(CVXOPTLinearProgramSolver):
        pass
elif linear_programming_backends["scipy"]:
    @BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(Number))))
    class LinearProgramSolver(SciPyLinearProgramSolver):
        pass
else:
    raise RuntimeError("No linear programming backends found")
