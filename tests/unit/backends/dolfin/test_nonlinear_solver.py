# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import dot, isclose
from numpy.linalg import norm as monitor_norm
import matplotlib
import matplotlib.pyplot as plt
from dolfin import (assemble, assign, derivative, DirichletBC, DOLFIN_EPS, dx, Expression, FunctionSpace, grad, inner,
                    IntervalMesh, pi, plot, project, TestFunction, TrialFunction)
from rbnics.backends.abstract import NonlinearProblemWrapper

"""
Solve
    - ((1 + u^2) u_x)_x = g,   x in [0, 2*pi]
      u = x,                   x on {0, 2*pi}
for g such that u = u_ex = x + sin(2*x)
"""


# ~~~ Sparse case ~~~ #
def _test_nonlinear_solver_sparse(callback_type):
    from dolfin import Function
    from rbnics.backends.dolfin import NonlinearSolver

    # Create mesh and define function space
    mesh = IntervalMesh(132, 0, 2 * pi)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 2 * pi)
    def boundary(x):
        return x[0] < 0 + DOLFIN_EPS or x[0] > 2 * pi - 10 * DOLFIN_EPS

    # Define exact solution
    exact_solution_expression = Expression("x[0] + sin(2*x[0])", element=V.ufl_element())
    exact_solution = project(exact_solution_expression, V)

    # Define variational problem
    du = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    g = Expression(
        "4 * sin(2 * x[0]) * (pow(x[0] + sin(2 * x[0]), 2) + 1)"
        + " - 2 * (x[0] + sin(2 * x[0])) * pow(2 * cos(2 * x[0]) + 1,"
        + " 2)",
        element=V.ufl_element())
    r = inner((1 + u**2) * grad(u), grad(v)) * dx - g * v * dx
    j = derivative(r, u, du)
    x = inner(du, v) * dx

    # Assemble inner product matrix
    X = assemble(x)

    # Define initial guess
    def initial_guess():
        initial_guess_expression = Expression("0.1 + 0.9 * x[0]", element=V.ufl_element())
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
    class ProblemWrapper(NonlinearProblemWrapper):
        # Residual function
        def residual_eval(self, solution):
            return callback(r)

        # Jacobian function
        def jacobian_eval(self, solution):
            return callback(j)

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

    # Solve the nonlinear problem
    problem_wrapper = ProblemWrapper()
    solution = u
    assign(solution, initial_guess())
    solver = NonlinearSolver(problem_wrapper, solution)
    solver.set_parameters({
        "linear_solver": "mumps",
        "maximum_iterations": 20,
        "report": True
    })
    solver.solve()

    # Compute the error
    error = Function(V)
    error.vector().add_local(+ solution.vector().get_local())
    error.vector().add_local(- exact_solution.vector().get_local())
    error.vector().apply("")
    error_norm = error.vector().inner(X * error.vector())
    print("Sparse error (" + callback_type + "):", error_norm)
    assert isclose(error_norm, 0., atol=1.e-5)
    return (error_norm, V, u, r, j, X, initial_guess, exact_solution)


# ~~~ Dense case ~~~ #
def _test_nonlinear_solver_dense(V, u, r, j, X, sparse_initial_guess, exact_solution):
    from rbnics.backends.online.numpy import Function, NonlinearSolver, Matrix, Vector

    # Define boundary condition
    x_to_dof = dict(zip(V.tabulate_dof_coordinates().flatten(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2 * pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)

    def dense_initial_guess():
        solution = Function(*exact_solution.vector().get_local().shape)
        solution.vector()[:] = sparse_initial_guess().vector().get_local()
        solution_array = solution.vector()
        solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
        return solution

    class ProblemWrapper(NonlinearProblemWrapper):
        # Residual and jacobian functions, reordering resulting matrix and vector
        # such that dof_0 and dof_2pi are in the first two rows/cols,
        # because the dense nonlinear solver has implicitly this assumption
        def residual_eval(self, solution):
            self._solution_from_dense_to_sparse(solution)
            residual_array = assemble(r).get_local()
            residual_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = residual_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            residual = Vector(*residual_array.shape)
            residual[:] = residual_array
            return residual

        def jacobian_eval(self, solution):
            self._solution_from_dense_to_sparse(solution)
            jacobian_array = assemble(j).array()
            jacobian_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = jacobian_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
            jacobian_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = jacobian_array[
                :, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            jacobian = Matrix(*jacobian_array.shape)
            jacobian[:, :] = jacobian_array
            return jacobian

        def bc_eval(self):
            if min_dof_0_2pi == dof_0:
                return (0., 2 * pi)
            else:
                return (2 * pi, 0.)

        # Define custom monitor to plot the solution
        def monitor(self, solution):
            if matplotlib.get_backend() != "agg":
                self._solution_from_dense_to_sparse(solution)
                plot(u, title="u")
                plt.show(block=False)
                plt.pause(1)
            else:
                print("||u|| = " + str(monitor_norm(solution.vector())))

        def _solution_from_dense_to_sparse(self, solution):
            solution_array = solution.vector()
            solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[
                [0, 1, min_dof_0_2pi, max_dof_0_2pi]]
            u.vector().zero()
            u.vector().add_local(solution_array.__array__())
            u.vector().apply("")
            solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]

    # Solve the nonlinear problem
    problem_wrapper = ProblemWrapper()
    solution = dense_initial_guess()
    solver = NonlinearSolver(problem_wrapper, solution)
    solver.set_parameters({
        "maximum_iterations": 20,
        "report": True
    })
    solver.solve()

    # Compute the error
    error = Function(*exact_solution.vector().get_local().shape)
    error.vector()[:] = exact_solution.vector().get_local()
    solution_array = solution.vector()
    solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    error.vector()[:] -= solution_array
    solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    error_norm = dot(error.vector(), dot(X.array(), error.vector()))
    print("Dense error:", error_norm)
    assert isclose(error_norm, 0., atol=1.e-5)
    return error_norm


# ~~~ Test function ~~~ #
def test_nonlinear_solver():
    (error_sparse_tensor_callbacks, V, u, r, j, X, sparse_initial_guess,
        exact_solution) = _test_nonlinear_solver_sparse("tensor callbacks")
    (error_sparse_form_callbacks, _, _, _, _, _, _, _) = _test_nonlinear_solver_sparse("form callbacks")
    assert isclose(error_sparse_tensor_callbacks, error_sparse_form_callbacks)
    if V.mesh().mpi_comm().size == 1:  # dense solver is not partitioned
        error_dense = _test_nonlinear_solver_dense(V, u, r, j, X, sparse_initial_guess, exact_solution)
        assert isclose(error_dense, error_sparse_tensor_callbacks)
        assert isclose(error_dense, error_sparse_form_callbacks)
