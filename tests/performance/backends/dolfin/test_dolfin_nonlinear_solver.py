# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import pytest
from numpy import isclose
from dolfin import (assemble, derivative, dx, Expression, Function, FunctionSpace, grad, inner, PETScOptions, project,
                    solve, TestFunction, TrialFunction, UnitSquareMesh)
from rbnics.backends import NonlinearSolver as FactoryNonlinearSolver
from rbnics.backends.abstract import NonlinearProblemWrapper
from rbnics.backends.dolfin import NonlinearSolver as DolfinNonlinearSolver
from test_dolfin_utils import RandomDolfinFunction

NonlinearSolver = None
AllNonlinearSolver = {"dolfin": DolfinNonlinearSolver, "factory": FactoryNonlinearSolver}

PETScOptions.set("snes_linesearch_type", "basic")


class Data(object):
    def __init__(self, Th, callback_type):
        # Create mesh and define function space
        mesh = UnitSquareMesh(Th, Th)
        self.V = FunctionSpace(mesh, "Lagrange", 1)
        # Define variational problem
        du = TrialFunction(self.V)
        v = TestFunction(self.V)
        self.u = Function(self.V)
        self.r = lambda u, g: inner(grad(u), grad(v)) * dx + inner(u + u**3, v) * dx - g * v * dx
        self.j = lambda u, r: derivative(r, u, du)
        # Define initial guess
        self.initial_guess_expression = Expression("0.1 + 0.9*x[0]*x[1]", element=self.V.ufl_element())
        # Define callback function depending on callback type
        assert callback_type in ("form callbacks", "tensor callbacks")
        if callback_type == "form callbacks":
            def callback(arg):
                return arg
        elif callback_type == "tensor callbacks":
            def callback(arg):
                return assemble(arg)
        self.callback_type = callback_type
        self.callback = callback

    def generate_random(self):
        # Generate random forcing
        g = RandomDolfinFunction(self.V)

        # Generate correspondingly residual and jacobian forms
        r = self.r(self.u, g)
        j = self.j(self.u, r)

        # Prepare problem wrapper
        class ProblemWrapper(NonlinearProblemWrapper):
            # Residual and jacobian functions
            def residual_eval(self_, solution):
                return self.callback(r)

            def jacobian_eval(self_, solution):
                return self.callback(j)

            # Define boundary condition
            def bc_eval(self_):
                return None

            # Empty solution monitor
            def monitor(self_, solution):
                pass

        problem_wrapper = ProblemWrapper()
        # Return
        return (r, j, problem_wrapper)

    def evaluate_builtin(self, r, j, problem_wrapper):
        project(self.initial_guess_expression, self.V, function=self.u)
        solve(
            r == 0, self.u, J=j,
            solver_parameters={
                "nonlinear_solver": "snes",
                "snes_solver": {
                    "linear_solver": "mumps",
                    "maximum_iterations": 20,
                    "relative_tolerance": 1e-9,
                    "absolute_tolerance": 1e-9,
                    "maximum_residual_evaluations": 10000,
                    "report": True
                }
            }
        )
        return self.u.copy(deepcopy=True)

    def evaluate_backend(self, r, j, problem_wrapper):
        project(self.initial_guess_expression, self.V, function=self.u)
        solver = NonlinearSolver(problem_wrapper, self.u)
        solver.set_parameters({
            "linear_solver": "mumps",
            "maximum_iterations": 20,
            "relative_tolerance": 1e-9,
            "absolute_tolerance": 1e-9,
            "report": True
        })
        solver.solve()
        return self.u.copy(deepcopy=True)

    def assert_backend(self, r, j, problem_wrapper, result_backend):
        result_builtin = self.evaluate_builtin(r, j, problem_wrapper)
        error = Function(self.V)
        error.vector().add_local(+ result_backend.vector().get_local())
        error.vector().add_local(- result_builtin.vector().get_local())
        error.vector().apply("add")
        relative_error = error.vector().norm("l2") / result_builtin.vector().norm("l2")
        assert isclose(relative_error, 0., atol=1e-12)


@pytest.mark.parametrize("Th", [2**i for i in range(3, 8)])
@pytest.mark.parametrize("callback_type", ["form callbacks", "tensor callbacks"])
@pytest.mark.parametrize("test_type", ["builtin"] + list(AllNonlinearSolver.keys()))
def test_dolfin_nonlinear_solver(Th, callback_type, test_type, benchmark):
    data = Data(Th, callback_type)
    print("Th = " + str(Th) + ", Nh = " + str(data.V.dim()))
    if test_type == "builtin":
        print("Testing " + test_type + ", callback_type = " + callback_type)
        benchmark(data.evaluate_builtin, setup=data.generate_random)
    else:
        print("Testing " + test_type + " backend" + ", callback_type = " + callback_type)
        global NonlinearSolver
        NonlinearSolver = AllNonlinearSolver[test_type]
        benchmark(data.evaluate_backend, setup=data.generate_random, teardown=data.assert_backend)
