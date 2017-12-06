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

from numpy import dot, isclose
from dolfin import assemble, DirichletBC, DOLFIN_EPS, dx, Expression, Function, FunctionSpace, grad, inner, IntervalMesh, pi, project, TestFunction, TrialFunction
from rbnics.backends.dolfin import LinearSolver as SparseLinearSolver
from rbnics.backends.online.numpy import Function as DenseFunction, LinearSolver as DenseLinearSolver, Matrix as DenseMatrix, Vector as DenseVector

"""
Solve
    - u_xx = g,   x in [0, 2*pi]
      u = x,      x on {0, 2*pi}
for g such that u = u_ex = x + sin(2*x)
"""

# ~~~ Sparse case ~~~ #
def _test_linear_solver_sparse(callback_type):
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
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Expression("4*sin(2*x[0])", element=V.ufl_element())
    a = inner(grad(u), grad(v))*dx
    f = g*v*dx
    x = inner(u, v)*dx

    # Assemble inner product matrix
    X = assemble(x)
    
    # Define boundary condition
    bc = [DirichletBC(V, exact_solution_expression, boundary)]
    
    # Defin solution
    sparse_solution = Function(V)
    
    # Define linear solver depending on callback type, and solve the problem
    assert callback_type in ("form callbacks", "tensor callbacks")
    if callback_type == "form callbacks":
        A = None
        F = None
        sparse_solver = SparseLinearSolver(a, sparse_solution, f, bc)
    elif callback_type == "tensor callbacks":
        A = assemble(a)
        F = assemble(f)
        sparse_solver = SparseLinearSolver(A, sparse_solution, F, bc)
    sparse_solver.solve()

    # Compute the error
    sparse_error = Function(V)
    sparse_error.vector().add_local(+ sparse_solution.vector().get_local())
    sparse_error.vector().add_local(- exact_solution.vector().get_local())
    sparse_error.vector().apply("")
    sparse_error_norm = sparse_error.vector().inner(X*sparse_error.vector())
    print("SparseLinearSolver error (" + callback_type + "):", sparse_error_norm)
    assert isclose(sparse_error_norm, 0., atol=1.e-5)
    return (sparse_error_norm, V, A, F, X, exact_solution)

# ~~~ Dense case ~~~ #
def _test_linear_solver_dense(V, A, F, X, exact_solution):
    # Define boundary condition
    x_to_dof = dict(zip(V.tabulate_dof_coordinates().flatten(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2*pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)
    if min_dof_0_2pi == dof_0:
        dense_bc = (0., 2*pi)
    else:
        dense_bc = (2*pi, 0.)
    
    # Reorder A and F such that dof_0 and dof_2pi are in the first two rows/cols,
    # because the dense linear solver has implicitly this assumption
    dense_A_array = A.array()
    dense_F_array = F.get_local()
    dense_A_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = dense_A_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
    dense_A_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_A_array[:, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    dense_F_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = dense_F_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]

    # Assemble matrix and vector
    dense_A = DenseMatrix(*dense_A_array.shape)
    dense_F = DenseVector(*dense_F_array.shape)
    dense_A[:, :] = dense_A_array
    dense_F[:] = dense_F_array
    
    # Solve the linear problem
    dense_solution = DenseFunction(*dense_F_array.shape)
    dense_solver = DenseLinearSolver(dense_A, dense_solution, dense_F, dense_bc)
    dense_solver.solve()
    dense_solution_array = dense_solution.vector()
    dense_solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = dense_solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    
    # Compute the error
    dense_error = DenseFunction(*dense_F_array.shape)
    dense_error.vector()[:] = exact_solution.vector().get_local()
    dense_error.vector()[:] -= dense_solution_array
    dense_error_norm = dot(dense_error.vector(), dot(X.array(), dense_error.vector()))
    print("DenseLinearSolver error:", dense_error_norm)
    assert isclose(dense_error_norm, 0., atol=1.e-5)
    return dense_error_norm
    
# ~~~ Test function ~~~ #
def test_linear_solver():
    (error_sparse_tensor_callbacks, V, A, F, X, exact_solution) = _test_linear_solver_sparse("tensor callbacks")
    (error_sparse_form_callbacks, _, _, _, _, _) = _test_linear_solver_sparse("form callbacks")
    assert isclose(error_sparse_tensor_callbacks, error_sparse_form_callbacks)
    if V.mesh().mpi_comm().size == 1: # dense solver is not partitioned
        error_dense = _test_linear_solver_dense(V, A, F, X, exact_solution)
        assert isclose(error_dense, error_sparse_tensor_callbacks)
        assert isclose(error_dense, error_sparse_form_callbacks)
