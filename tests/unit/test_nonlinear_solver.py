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

from numpy import asarray, isclose
from dolfin import *
from rbnics.backends.abstract import NonlinearProblemWrapper
from rbnics.backends.dolfin import NonlinearSolver as SparseNonlinearSolver
from rbnics.backends.online.numpy import Function as DenseFunction, NonlinearSolver as DenseNonlinearSolver, Matrix as DenseMatrix, Vector as DenseVector

"""
Solve
    - ((1 + u^2) u_x)_x = g,   x in [0, 2*pi]
      u = x,                   x on {0, 2*pi}
for g such that u = u_ex = x + sin(2*x)
"""

# Create mesh and define function space
mesh = IntervalMesh(132, 0, 2*pi)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 2*pi)
def boundary(x):
    return x[0] < 0 + DOLFIN_EPS or x[0] > 2*pi - 10*DOLFIN_EPS
    
# Define exact solution
exact_solution_expression = Expression("x[0] + sin(2*x[0])", element=V.ufl_element())
exact_solution = project(exact_solution_expression, V)

# Define variational problem
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)
g = Expression("4*sin(2*x[0])*(pow(x[0]+sin(2*x[0]), 2)+1)-2*(x[0]+sin(2*x[0]))*pow(2*cos(2*x[0])+1, 2)", element=V.ufl_element())
r = inner((1+u**2)*grad(u), grad(v))*dx - g*v*dx
j = derivative(r, u, du)
x = inner(du, v)*dx

# Assemble inner product matrix
X = assemble(x)

# ~~~ Sparse case ~~~ #
# Define boundary condition
bc = [DirichletBC(V, exact_solution_expression, boundary)]

# Define initial guess
def sparse_initial_guess():
    initial_guess_expression = Expression("0.1 + 0.9*x[0]", element=V.ufl_element())
    return project(initial_guess_expression, V)

# ::: Callbacks return forms :: #
class SparseFormProblemWrapper(NonlinearProblemWrapper):
    # Residual and jacobian functions
    def residual_eval(self, solution):
        return r
    def jacobian_eval(self, solution):
        return j
        
    # Define boundary condition
    def bc_eval(self):
        return bc
    
# Solve the nonlinear problem
sparse_form_problem_wrapper = SparseFormProblemWrapper()
sparse_form_solution = u
assign(sparse_form_solution, sparse_initial_guess())
sparse_form_solver = SparseNonlinearSolver(sparse_form_problem_wrapper, sparse_form_solution)
sparse_form_solver.set_parameters({
    "linear_solver": "mumps",
    "maximum_iterations": 20,
    "report": True,
    "error_on_nonconvergence": True
})
sparse_form_solver.solve()

# Compute the error
sparse_form_error = Function(V)
sparse_form_error.vector().add_local(+ sparse_form_solution.vector().array())
sparse_form_error.vector().add_local(- exact_solution.vector().array())
sparse_form_error.vector().apply("")
sparse_form_error_norm = sparse_form_error.vector().inner(X*sparse_form_error.vector())
print("SparseNonlinearSolver error (form callbacks):", sparse_form_error_norm)
assert isclose(sparse_form_error_norm, 0., atol=1.e-5)

# ::: Callbacks return tensors :: #
class SparseTensorProblemWrapper(NonlinearProblemWrapper):
    # Residual and jacobian functions
    def residual_eval(self, solution):
        return assemble(r)
    def jacobian_eval(self, solution):
        return assemble(j)
        
    # Define boundary condition
    def bc_eval(self):
        return bc
    
# Solve the nonlinear problem
sparse_tensor_problem_wrapper = SparseTensorProblemWrapper()
sparse_tensor_solution = u
assign(sparse_tensor_solution, sparse_initial_guess())
sparse_tensor_solver = SparseNonlinearSolver(sparse_tensor_problem_wrapper, sparse_tensor_solution)
sparse_tensor_solver.set_parameters({
    "linear_solver": "mumps",
    "maximum_iterations": 20,
    "report": True,
    "error_on_nonconvergence": True
})
sparse_tensor_solver.solve()

# Compute the error
sparse_tensor_error = Function(V)
sparse_tensor_error.vector().add_local(+ sparse_tensor_solution.vector().array())
sparse_tensor_error.vector().add_local(- exact_solution.vector().array())
sparse_tensor_error.vector().apply("")
sparse_tensor_error_norm = sparse_tensor_error.vector().inner(X*sparse_tensor_error.vector())
print("SparseNonlinearSolver error (tensor callbacks):", sparse_tensor_error_norm)
assert isclose(sparse_tensor_error_norm, 0., atol=1.e-5)
assert isclose(sparse_tensor_error_norm, sparse_form_error_norm)

# ~~~ Dense case ~~~ #
if mesh.mpi_comm().size == 1: # dense solver is not partitioned
    # Define boundary condition
    x_to_dof = dict(zip(V.tabulate_dof_coordinates(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2*pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)
    
    def dense_initial_guess():
        sparse_function = sparse_initial_guess()
        dense_solution = DenseFunction(*sparse_function.vector().array().shape)
        dense_solution.vector()[:] = sparse_function.vector().array().reshape((-1, 1))
        dense_solution_array = dense_solution.vector()
        dense_solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
        return dense_solution
    
    class DenseProblemWrapper(NonlinearProblemWrapper):
        # Residual and jacobian functions, reordering resulting matrix and vector
        # such that dof_0 and dof_2pi are in the first two rows/cols,
        # because the dense nonlinear solver has implicitly this assumption
        def residual_eval(self, solution):
            self._solution_from_dense_to_sparse(solution)
            sparse_residual = assemble(r)
            dense_residual_array = sparse_residual.array()
            dense_residual_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_residual_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            dense_residual = DenseVector(*dense_residual_array.shape)
            dense_residual[:] = dense_residual_array.reshape((-1, 1))
            return dense_residual
            
        def jacobian_eval(self, solution):
            self._solution_from_dense_to_sparse(solution)
            sparse_jacobian = assemble(j)
            dense_jacobian_array = sparse_jacobian.array()
            dense_jacobian_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = dense_jacobian_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
            dense_jacobian_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_jacobian_array[:, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            dense_jacobian = DenseMatrix(*dense_jacobian_array.shape)
            dense_jacobian[:, :] = dense_jacobian_array
            return dense_jacobian
            
        def bc_eval(self):
            if min_dof_0_2pi == dof_0:
                return (0., 2*pi)
            else:
                return (2*pi, 0.)
            
        def _solution_from_dense_to_sparse(self, solution):
            solution_array = asarray(solution.vector()).reshape(-1)
            solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
            u.vector().zero()
            u.vector().add_local(solution_array)
            u.vector().apply("")
            solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
        
    # Solve the nonlinear problem
    dense_problem_wrapper = DenseProblemWrapper()
    dense_solution = dense_initial_guess()
    dense_solver = DenseNonlinearSolver(dense_problem_wrapper, dense_solution)
    dense_solver.set_parameters({
        "maximum_iterations": 20,
        "report": True
    })
    dense_solver.solve()
    dense_solution_array = dense_solution.vector()
    dense_solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = dense_solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    
    # Compute the error
    dense_error = DenseFunction(*exact_solution.vector().array().shape)
    dense_error.vector()[:] = exact_solution.vector().array().reshape((-1, 1))
    dense_error.vector()[:] -= dense_solution_array
    dense_error_norm = dense_error.vector().T.dot(X.array().dot(dense_error.vector()))
    assert dense_error_norm.shape == (1, 1)
    dense_error_norm = dense_error_norm[0, 0]
    print("DenseNonlinearSolver error:", dense_error_norm)
    assert isclose(dense_error_norm, 0., atol=1.e-5)
else:
    print("DenseNonlinearSolver error: skipped in parallel")
    

