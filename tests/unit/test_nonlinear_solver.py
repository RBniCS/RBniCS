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

from numpy import dot, isclose
from dolfin import assemble, assign, derivative, DirichletBC, DOLFIN_EPS, dx, Expression, Function, FunctionSpace, grad, inner, IntervalMesh, pi, project, TestFunction, TrialFunction
from rbnics.backends.abstract import NonlinearProblemWrapper
from rbnics.backends.dolfin import NonlinearSolver as SparseNonlinearSolver
from rbnics.backends.online.numpy import Function as DenseFunction, NonlinearSolver as DenseNonlinearSolver, Matrix as DenseMatrix, Vector as DenseVector

"""
Solve
    - ((1 + u^2) u_x)_x = g,   x in [0, 2*pi]
      u = x,                   x on {0, 2*pi}
for g such that u = u_ex = x + sin(2*x)
"""

# ~~~ Sparse case ~~~ #
def _test_nonlinear_solver_sparse(callback_type):
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
    
    # Define initial guess
    def sparse_initial_guess():
        initial_guess_expression = Expression("0.1 + 0.9*x[0]", element=V.ufl_element())
        return project(initial_guess_expression, V)

    # Define boundary condition
    bc = [DirichletBC(V, exact_solution_expression, boundary)]
    
    # Define callback function depending on callback type
    assert callback_type in ("form callbacks", "tensor callbacks")
    if callback_type == "form callbacks":
        def callback(arg):
            return arg
    elif callback_type == "tensor callbacks":
        def callback(arg):
            return assemble(arg)
    
    # Define problem wrapper
    class SparseFormProblemWrapper(NonlinearProblemWrapper):
        # Residual function
        def residual_eval(self, solution):
            return callback(r)
            
        # Jacobian function
        def jacobian_eval(self, solution):
            return callback(j)
            
        # Define boundary condition
        def bc_eval(self):
            return bc
        
    # Solve the nonlinear problem
    sparse_problem_wrapper = SparseFormProblemWrapper()
    sparse_solution = u
    assign(sparse_solution, sparse_initial_guess())
    sparse_solver = SparseNonlinearSolver(sparse_problem_wrapper, sparse_solution)
    sparse_solver.set_parameters({
        "linear_solver": "mumps",
        "maximum_iterations": 20,
        "report": True
    })
    sparse_solver.solve()

    # Compute the error
    sparse_error = Function(V)
    sparse_error.vector().add_local(+ sparse_solution.vector().get_local())
    sparse_error.vector().add_local(- exact_solution.vector().get_local())
    sparse_error.vector().apply("")
    sparse_error_norm = sparse_error.vector().inner(X*sparse_error.vector())
    print("SparseNonlinearSolver error (" + callback_type + "):", sparse_error_norm)
    assert isclose(sparse_error_norm, 0., atol=1.e-5)
    return (sparse_error_norm, V, u, r, j, X, sparse_initial_guess, exact_solution)

# ~~~ Dense case ~~~ #
def _test_nonlinear_solver_dense(V, u, r, j, X, sparse_initial_guess, exact_solution):
    # Define boundary condition
    x_to_dof = dict(zip(V.tabulate_dof_coordinates().flatten(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2*pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)
    
    def dense_initial_guess():
        sparse_function = sparse_initial_guess()
        dense_solution = DenseFunction(*sparse_function.vector().get_local().shape)
        dense_solution.vector()[:] = sparse_function.vector().get_local()
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
            dense_residual_array = sparse_residual.get_local()
            dense_residual_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_residual_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            dense_residual = DenseVector(*dense_residual_array.shape)
            dense_residual[:] = dense_residual_array
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
            solution_array = solution.vector()
            solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
            u.vector().zero()
            u.vector().add_local(solution_array.__array__())
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
    dense_error = DenseFunction(*exact_solution.vector().get_local().shape)
    dense_error.vector()[:] = exact_solution.vector().get_local()
    dense_error.vector()[:] -= dense_solution_array
    dense_error_norm = dot(dense_error.vector(), dot(X.array(), dense_error.vector()))
    print("DenseNonlinearSolver error:", dense_error_norm)
    assert isclose(dense_error_norm, 0., atol=1.e-5)
    return dense_error_norm
    
# ~~~ Test function ~~~ #
def test_nonlinear_solver():
    (error_sparse_tensor_callbacks, V, u, r, j, X, sparse_initial_guess, exact_solution) = _test_nonlinear_solver_sparse("tensor callbacks")
    (error_sparse_form_callbacks, _, _, _, _, _, _, _) = _test_nonlinear_solver_sparse("form callbacks")
    assert isclose(error_sparse_tensor_callbacks, error_sparse_form_callbacks)
    if V.mesh().mpi_comm().size == 1: # dense solver is not partitioned
        error_dense = _test_nonlinear_solver_dense(V, u, r, j, X, sparse_initial_guess, exact_solution)
        assert isclose(error_dense, error_sparse_tensor_callbacks)
        assert isclose(error_dense, error_sparse_form_callbacks)
