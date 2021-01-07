# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import dot, isclose
from numpy.linalg import norm as monitor_norm
import matplotlib
import matplotlib.pyplot as plt
from dolfin import (assemble, DirichletBC, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner, IntervalMesh, pi,
                    plot, project, TestFunction, TrialFunction)
from rbnics.backends.abstract import LinearProblemWrapper

"""
Solve
    - u_xx = g,   x in [0, 2*pi]
      u = x,      x on {0, 2*pi}
for g such that u = u_ex = x + sin(2*x)
"""


# ~~~ Sparse case ~~~ #
def _test_linear_solver_sparse(callback_type):
    from dolfin import Function
    from rbnics.backends.dolfin import LinearSolver

    # Create mesh and define function space
    mesh = IntervalMesh(132, 0, 2 * pi)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 2 * pi)
    def boundary(x):
        return x[0] < 0 + DOLFIN_EPS or x[0] > 2 * pi - 10 * DOLFIN_EPS

    # Define exact solution
    exact_solution_expression = Expression("x[0] + sin(2 * x[0])", element=V.ufl_element())
    exact_solution = project(exact_solution_expression, V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    g = Expression("4 * sin(2 * x[0])", element=V.ufl_element())
    a = inner(grad(u), grad(v)) * dx
    f = g * v * dx
    x = inner(u, v) * dx

    # Assemble inner product matrix
    X = assemble(x)

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
    class ProblemWrapper(LinearProblemWrapper):
        # Vector function
        def vector_eval(self):
            return callback(f)

        # Matrix function
        def matrix_eval(self):
            return callback(a)

        # Define boundary condition
        def bc_eval(self):
            return bc

        # Define custom monitor to plot the solution
        def monitor(self, solution):
            if matplotlib.get_backend() != "agg":
                plot(solution, title="u")
                plt.show(block=False)
                plt.pause(1)
            else:
                print("||u|| = " + str(solution.vector().norm("l2")))

    # Solve the linear problem
    problem_wrapper = ProblemWrapper()
    solution = Function(V)
    solver = LinearSolver(problem_wrapper, solution)
    solver.solve()

    # Compute the error
    error = Function(V)
    error.vector().add_local(+ solution.vector().get_local())
    error.vector().add_local(- exact_solution.vector().get_local())
    error.vector().apply("")
    error_norm = error.vector().inner(X * error.vector())
    print("Sparse error (" + callback_type + "):", error_norm)
    assert isclose(error_norm, 0., atol=1.e-5)
    return (error_norm, V, a, f, X, exact_solution)


# ~~~ Dense case ~~~ #
def _test_linear_solver_dense(V, a, f, X, exact_solution):
    from rbnics.backends.online.numpy import Function, LinearSolver, Matrix, Vector

    # Define boundary condition
    x_to_dof = dict(zip(V.tabulate_dof_coordinates().flatten(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2 * pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)

    class ProblemWrapper(LinearProblemWrapper):
        # Vector and matrix functions, reordering resulting matrix and vector
        # such that dof_0 and dof_2pi are in the first two rows/cols,
        # because the dense nonlinear solver has implicitly this assumption
        def vector_eval(self):
            F_array = assemble(f).get_local()
            F_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = F_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            F = Vector(*F_array.shape)
            F[:] = F_array
            return F

        def matrix_eval(self):
            A_array = assemble(a).array()
            A_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = A_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
            A_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = A_array[:, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            A = Matrix(*A_array.shape)
            A[:, :] = A_array
            return A

        def bc_eval(self):
            if min_dof_0_2pi == dof_0:
                return (0., 2 * pi)
            else:
                return (2 * pi, 0.)

        # Define custom monitor to plot the solution
        def monitor(self, solution):
            if matplotlib.get_backend() != "agg":
                solution_array = solution.vector()
                solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[
                    [0, 1, min_dof_0_2pi, max_dof_0_2pi]]
                plt.plot(x_to_dof.keys(), solution_array)
                solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[
                    [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
                plt.show(block=False)
                plt.pause(1)
            else:
                print("||u|| = " + str(monitor_norm(solution.vector())))

    # Solve the linear problem
    problem_wrapper = ProblemWrapper()
    solution = Function(*exact_solution.vector().get_local().shape)
    solver = LinearSolver(problem_wrapper, solution)
    solver.solve()
    solution_array = solution.vector()

    # Compute the error
    error = Function(*exact_solution.vector().get_local().shape)
    error.vector()[:] = exact_solution.vector().get_local()
    solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    error.vector()[:] -= solution_array
    solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    error_norm = dot(error.vector(), dot(X.array(), error.vector()))
    print("Dense error:", error_norm)
    assert isclose(error_norm, 0., atol=1.e-5)
    return error_norm


# ~~~ Test function ~~~ #
def test_linear_solver():
    (error_sparse_tensor_callbacks, V, a, f, X, exact_solution) = _test_linear_solver_sparse("tensor callbacks")
    (error_sparse_form_callbacks, _, _, _, _, _) = _test_linear_solver_sparse("form callbacks")
    assert isclose(error_sparse_tensor_callbacks, error_sparse_form_callbacks)
    if V.mesh().mpi_comm().size == 1:  # dense solver is not partitioned
        error_dense = _test_linear_solver_dense(V, a, f, X, exact_solution)
        assert isclose(error_dense, error_sparse_tensor_callbacks)
        assert isclose(error_dense, error_sparse_form_callbacks)
