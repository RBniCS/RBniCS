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

linear_programming_backends = {
    "cvxopt": None,
    "python-glpk": None,
    "scipy": None
}

try:
    import cvxopt
except ImportError:
    linear_programming_backends["cvxopt"] = False
else:
    linear_programming_backends["cvxopt"] = True
    # Set global options
    cvxopt.solvers.options["show_progress"] = False
    cvxopt.solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}

try:
    import glpk
except ImportError:
    linear_programming_backends["python-glpk"] = False
else:
    linear_programming_backends["python-glpk"] = True

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
            result = cvxopt.solvers.lp(self.cost, self.inequality_constraints_matrix, self.inequality_constraints_vector, solver="glpk")
            if result["status"] != "optimal":
                raise Error("Linear program solver reports convergence failure with reason", result["status"])
            else:
                return result["primal objective"]
    
if linear_programming_backends["python-glpk"]:
    class PythonGLPKLinearProgramSolver(AbstractLinearProgramSolver):
        def __init__(self, cost, inequality_constraints_matrix, inequality_constraints_vector, bounds):
            self.cost = cost
            self.inequality_constraints_matrix = inequality_constraints_matrix
            self.inequality_constraints_vector = inequality_constraints_vector
            self.bounds = bounds
            
        def solve(self):
            lp = glpk.glp_create_prob()
            glpk.glp_set_obj_dir(lp, glpk.GLP_MIN)
            
            # A. Linear program unknowns: Q variables, y_1, ..., y_Q
            Q = len(self.cost)
            glpk.glp_add_cols(lp, Q)
            
            # B. Range: constrain the variables to be in the bounding box (note: GLPK indexing starts from 1)
            assert len(self.bounds) == Q
            for (q, bounds_q) in enumerate(self.bounds):
                assert bounds_q[0] <= bounds_q[1]
                if bounds_q[0] < bounds_q[1]: # the usual case
                    glpk.glp_set_col_bnds(lp, q + 1, glpk.GLP_DB, bounds_q[0], bounds_q[1])
                elif bounds_q[0] == bounds_q[1]: # unlikely, but possible
                    glpk.glp_set_col_bnds(lp, q + 1, glpk.GLP_FX, bounds_q[0], bounds_q[1])
                else: # there is something wrong in the bounding box: set as unconstrained variable
                    raise AssertionError("bounds_min > bounds_max")
                    
            # C. Add inequality constraints
            assert self.inequality_constraints_vector.size == self.inequality_constraints_matrix.shape[0]
            assert Q == self.inequality_constraints_matrix.shape[1]
            glpk.glp_add_rows(lp, self.inequality_constraints_vector.size)
            array_size = self.inequality_constraints_matrix.shape[0]*self.inequality_constraints_matrix.shape[1]
            matrix_row_index = glpk.intArray(array_size + 1) # + 1 since GLPK indexing starts from 1
            matrix_column_index = glpk.intArray(array_size + 1)
            matrix_content = glpk.doubleArray(array_size + 1)
            glpk_container_size = 0
            for j in range(self.inequality_constraints_matrix.shape[0]):
                # Assemble the LHS of the constraint
                for q in range(self.inequality_constraints_matrix.shape[1]):
                    matrix_row_index[glpk_container_size + 1] = int(j + 1)
                    matrix_column_index[glpk_container_size + 1] = int(q + 1)
                    matrix_content[glpk_container_size + 1] = self.inequality_constraints_matrix[j, q]
                    glpk_container_size += 1
                    
                # Load the RHS of the constraint
                glpk.glp_set_row_bnds(lp, j + 1, glpk.GLP_LO, self.inequality_constraints_vector[j], 0.)
                
            # Load the assembled LHS
            glpk.glp_load_matrix(lp, array_size, matrix_row_index, matrix_column_index, matrix_content)
            
            # D. Set cost function coefficients
            for q in range(Q):
                glpk.glp_set_obj_coef(lp, q + 1, self.cost[q])
                
            # E. Solve
            options = glpk.glp_smcp()
            glpk.glp_init_smcp(options)
            options.msg_lev = glpk.GLP_MSG_ERR
            options.meth = glpk.GLP_DUAL
            glpk.glp_simplex(lp, options)
            min_f = glpk.glp_get_obj_val(lp)
            
            # F. Clean up
            glpk.glp_delete_prob(lp)
            
            return min_f
            
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
            
if linear_programming_backends["cvxopt"]:
    @BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(float))))
    class LinearProgramSolver(CVXOPTLinearProgramSolver):
        pass
elif linear_programming_backends["python-glpk"]:
    @BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(float))))
    class LinearProgramSolver(PythonGLPKLinearProgramSolver):
        pass
elif linear_programming_backends["scipy"]:
    @BackendFor("common", inputs=(numpy_vector, numpy_matrix, numpy_vector, list_of(tuple_of(float))))
    class LinearProgramSolver(SciPyLinearProgramSolver):
        pass
else:
    raise RuntimeError("No linear programming backends found")
