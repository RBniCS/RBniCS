# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import dot, isclose
from numpy.linalg import norm as monitor_norm
from dolfin import (assemble, Constant, derivative, DirichletBC, DOLFIN_EPS, dx, Expression, FunctionSpace, grad,
                    inner, IntervalMesh, PETScOptions, pi, plot, project, sin, TestFunction, TrialFunction)
import matplotlib
import matplotlib.pyplot as plt
from rbnics.backends.abstract import TimeDependentProblemWrapper
from rbnics.backends.online.numpy.time_stepping import has_IDA

# Additional command line options for PETSc TS
PETScOptions.set("ts_bdf_order", "3")
PETScOptions.set("ts_bdf_adapt", "true")

"""
Solve
    u_t - ((1 + u^2) u_x)_x = g,   (t, x) in [0, 1] x [0, 2*pi]
    u = sin(t),                    (t, x) in [0, 1] x {0, 2*pi}
    u = sin(x),                    (t, x) in {0}    x [0, 2*pi]
for g such that u = u_ex = sin(x+t)
"""


# ~~~ Sparse case ~~~ #
def _test_time_stepping_2_sparse(callback_type, integrator_type):
    from dolfin import Function
    from rbnics.backends.dolfin import TimeStepping

    # Create mesh and define function space
    mesh = IntervalMesh(132, 0, 2 * pi)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define Dirichlet boundary (x = 0 or x = 2*pi)
    def boundary(x):
        return x[0] < 0 + DOLFIN_EPS or x[0] > 2 * pi - 10 * DOLFIN_EPS

    # Define time step
    dt = 0.01
    monitor_dt = 0.02
    T = 1.

    # Define exact solution
    exact_solution_expression = Expression("sin(x[0]+t)", t=0, element=V.ufl_element())
    # ... and interpolate it at the final time
    exact_solution_expression.t = T
    exact_solution = project(exact_solution_expression, V)

    # Define exact solution dot
    exact_solution_dot_expression = Expression("cos(x[0]+t)", t=0, element=V.ufl_element())
    # ... and interpolate it at the final time
    exact_solution_dot_expression.t = T
    exact_solution_dot = project(exact_solution_dot_expression, V)

    # Define variational problem
    du = TrialFunction(V)
    du_dot = TrialFunction(V)
    v = TestFunction(V)
    u = Function(V)
    u_dot = Function(V)
    g = Expression("5. / 4. * sin(t + x[0]) - 3. / 4. * sin(3 * (t + x[0])) + cos(t + x[0])", t=0.,
                   element=V.ufl_element())
    r_u = inner((1 + u**2) * grad(u), grad(v)) * dx
    j_u = derivative(r_u, u, du)
    r_u_dot = inner(u_dot, v) * dx
    j_u_dot = derivative(r_u_dot, u_dot, du_dot)
    r = r_u_dot + r_u - g * v * dx
    x = inner(du, v) * dx

    def bc(t):
        exact_solution_expression.t = t
        return [DirichletBC(V, exact_solution_expression, boundary)]

    # Assemble inner product matrix
    X = assemble(x)

    # Define callback function depending on callback type
    assert callback_type in ("form callbacks", "tensor callbacks")
    if callback_type == "form callbacks":
        def callback(arg):
            return arg
    elif callback_type == "tensor callbacks":
        def callback(arg):
            return assemble(arg)

    # Define problem wrapper
    class ProblemWrapper(TimeDependentProblemWrapper):
        # Residual and jacobian functions
        def residual_eval(self, t, solution, solution_dot):
            g.t = t
            return callback(r)

        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            return callback(Constant(solution_dot_coefficient) * j_u_dot + j_u)

        # Define boundary condition
        def bc_eval(self, t):
            return bc(t)

        # Define initial condition
        def ic_eval(self):
            exact_solution_expression.t = 0.
            return project(exact_solution_expression, V)

        # Define custom monitor to plot the solution
        def monitor(self, t, solution, solution_dot):
            assert isclose(round(t / monitor_dt), t / monitor_dt)
            if matplotlib.get_backend() != "agg":
                plt.subplot(1, 2, 1).clear()
                plot(solution, title="u at t = " + str(t))
                plt.subplot(1, 2, 2).clear()
                plot(solution_dot, title="u_dot at t = " + str(t))
                plt.show(block=False)
                plt.pause(DOLFIN_EPS)
            else:
                print("||u|| at t = " + str(t) + ": " + str(solution.vector().norm("l2")))
                print("||u_dot|| at t = " + str(t) + ": " + str(solution_dot.vector().norm("l2")))

    # Solve the time dependent problem
    problem_wrapper = ProblemWrapper()
    (solution, solution_dot) = (u, u_dot)
    solver = TimeStepping(problem_wrapper, solution, solution_dot)
    solver.set_parameters({
        "initial_time": 0.0,
        "time_step_size": dt,
        "monitor": {
            "time_step_size": monitor_dt,
        },
        "final_time": T,
        "exact_final_time": "stepover",
        "integrator_type": integrator_type,
        "problem_type": "nonlinear",
        "snes_solver": {
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "report": True
        },
        "report": True
    })
    solver.solve()

    # Compute the error at the final time
    error = Function(V)
    error.vector().add_local(+ solution.vector().get_local())
    error.vector().add_local(- exact_solution.vector().get_local())
    error.vector().apply("")
    error_norm = error.vector().inner(X * error.vector())
    error_dot = Function(V)
    error_dot.vector().add_local(+ solution_dot.vector().get_local())
    error_dot.vector().add_local(- exact_solution_dot.vector().get_local())
    error_dot.vector().apply("")
    error_dot_norm = error_dot.vector().inner(X * error_dot.vector())
    print("Sparse error (" + callback_type + ", " + integrator_type + "):", error_norm, error_dot_norm)
    assert isclose(error_norm, 0., atol=1.e-4)
    assert isclose(error_dot_norm, 0., atol=1.e-4)
    return ((error_norm, error_dot_norm), V, dt, monitor_dt, T, u, u_dot, g, r, j_u, j_u_dot, X,
            exact_solution_expression, exact_solution, exact_solution_dot)


# ~~~ Dense case ~~~ #
def _test_time_stepping_2_dense(
        integrator_type, V, dt, monitor_dt, T, u, u_dot, g, r, j_u, j_u_dot, X,
        exact_solution_expression, exact_solution, exact_solution_dot):
    from rbnics.backends.online.numpy import Function, Matrix, TimeStepping, Vector

    x_to_dof = dict(zip(V.tabulate_dof_coordinates().flatten(), V.dofmap().dofs()))
    dof_0 = x_to_dof[0.]
    dof_2pi = x_to_dof[2 * pi]
    min_dof_0_2pi = min(dof_0, dof_2pi)
    max_dof_0_2pi = max(dof_0, dof_2pi)

    # Define problem wrapper
    class ProblemWrapper(TimeDependentProblemWrapper):
        # Residual and jacobian functions, reordering resulting matrix and vector
        # such that dof_0 and dof_2pi are in the first two rows/cols,
        # because the dense time stepping solver has implicitly this assumption
        def residual_eval(self, t, solution, solution_dot):
            self._solution_from_dense_to_sparse(solution, u)
            self._solution_from_dense_to_sparse(solution_dot, u_dot)
            g.t = t
            residual_array = assemble(r).get_local()
            residual_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = residual_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            residual = Vector(*residual_array.shape)
            residual[:] = residual_array
            return residual

        def jacobian_eval(self, t, solution, solution_dot, solution_dot_coefficient):
            self._solution_from_dense_to_sparse(solution, u)
            self._solution_from_dense_to_sparse(solution_dot, u_dot)
            jacobian_array = assemble(Constant(solution_dot_coefficient) * j_u_dot + j_u).array()
            jacobian_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi], :] = jacobian_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1], :]
            jacobian_array[:, [0, 1, min_dof_0_2pi, max_dof_0_2pi]] = jacobian_array[
                :, [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            jacobian = Matrix(*jacobian_array.shape)
            jacobian[:, :] = jacobian_array
            return jacobian

        # Define boundary condition
        def bc_eval(self, t):
            return (sin(t), sin(t))

        # Define initial condition
        def ic_eval(self):
            exact_solution_expression.t = 0.
            solution = Function(*exact_solution.vector().get_local().shape)
            solution.vector()[:] = project(exact_solution_expression, V).vector().get_local()
            solution_array = solution.vector()
            solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
            return solution

        # Define custom monitor to plot the solution
        def monitor(self, t, solution, solution_dot):
            assert isclose(round(t / monitor_dt), t / monitor_dt)
            if matplotlib.get_backend() != "agg":
                self._solution_from_dense_to_sparse(solution, u)
                self._solution_from_dense_to_sparse(solution_dot, u_dot)
                plt.subplot(1, 2, 1).clear()
                plot(u, title="u at t = " + str(t))
                plt.subplot(1, 2, 2).clear()
                plot(u_dot, title="u_dot at t = " + str(t))
                plt.show(block=False)
                plt.pause(DOLFIN_EPS)
            else:
                print("||u|| at t = " + str(t) + ": " + str(monitor_norm(solution.vector())))
                print("||u_dot|| at t = " + str(t) + ": " + str(monitor_norm(solution_dot.vector())))

        def _solution_from_dense_to_sparse(self, solution, u):
            solution_array = solution.vector()
            solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[
                [0, 1, min_dof_0_2pi, max_dof_0_2pi]]
            u.vector().zero()
            u.vector().add_local(solution_array.__array__())
            u.vector().apply("")
            solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[
                [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]

    # Solve the time dependent problem
    problem_wrapper = ProblemWrapper()
    shape = exact_solution.vector().get_local().shape
    (solution, solution_dot) = (Function(*shape), Function(*shape))
    solver = TimeStepping(problem_wrapper, solution, solution_dot)
    solver_parameters = {
        "initial_time": 0.0,
        "time_step_size": dt,
        "monitor": {
            "time_step_size": monitor_dt,
        },
        "final_time": T,
        "integrator_type": integrator_type,
        "problem_type": "nonlinear",
        "report": True
    }
    if integrator_type != "ida":
        solver_parameters.update({
            "nonlinear_solver": {
                "maximum_iterations": 20,
                "report": True
            }
        })
    solver.set_parameters(solver_parameters)
    solver.solve()

    # Compute the error at the final time
    error = Function(*exact_solution.vector().get_local().shape)
    error.vector()[:] = exact_solution.vector().get_local()
    solution_array = solution.vector()
    solution_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_array[
        [0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    error.vector()[:] -= solution_array
    solution_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_array[
        [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    error_norm = dot(error.vector(), dot(X.array(), error.vector()))
    error_dot = Function(*exact_solution_dot.vector().get_local().shape)
    error_dot.vector()[:] = exact_solution_dot.vector().get_local()
    solution_dot_array = solution_dot.vector()
    solution_dot_array[[min_dof_0_2pi, max_dof_0_2pi, 0, 1]] = solution_dot_array[
        [0, 1, min_dof_0_2pi, max_dof_0_2pi]]
    error_dot.vector()[:] -= solution_dot_array
    solution_dot_array[[0, 1, min_dof_0_2pi, max_dof_0_2pi]] = solution_dot_array[
        [min_dof_0_2pi, max_dof_0_2pi, 0, 1]]
    error_dot_norm = dot(error_dot.vector(), dot(X.array(), error_dot.vector()))
    print("Dense error (" + integrator_type + "):", error_norm, error_dot_norm)
    assert isclose(error_norm, 0., atol=1.e-4)
    assert isclose(error_dot_norm, 0., atol=1.e-4)
    return (error_norm, error_dot_norm)


# ~~~ Test function ~~~ #
def test_time_stepping_2():
    (error_sparse_tensor_callbacks_beuler, V, dt, monitor_dt, T, u, u_dot, g, r, j_u, j_u_dot, X,
        exact_solution_expression, exact_solution, exact_solution_dot) = _test_time_stepping_2_sparse(
            "tensor callbacks", "beuler")
    (error_sparse_form_callbacks_beuler, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = _test_time_stepping_2_sparse(
        "form callbacks", "beuler")
    assert isclose(error_sparse_tensor_callbacks_beuler, error_sparse_form_callbacks_beuler).all()
    (error_sparse_tensor_callbacks_bdf, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = _test_time_stepping_2_sparse(
        "tensor callbacks", "bdf")
    (error_sparse_form_callbacks_bdf, _, _, _, _, _, _, _, _, _, _, _, _, _, _) = _test_time_stepping_2_sparse(
        "form callbacks", "bdf")
    assert isclose(error_sparse_tensor_callbacks_bdf, error_sparse_form_callbacks_bdf).all()
    if V.mesh().mpi_comm().size == 1:  # dense solver is not partitioned
        error_dense_beuler = _test_time_stepping_2_dense(
            "beuler", V, dt, monitor_dt, T, u, u_dot, g, r, j_u, j_u_dot, X,
            exact_solution_expression, exact_solution, exact_solution_dot)
        assert isclose(error_dense_beuler, error_sparse_tensor_callbacks_beuler).all()
        assert isclose(error_dense_beuler, error_sparse_form_callbacks_beuler).all()
        if has_IDA:
            _test_time_stepping_2_dense(
                "ida", V, dt, monitor_dt, T, u, u_dot, g, r, j_u, j_u_dot, X,
                exact_solution_expression, exact_solution, exact_solution_dot)
