# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import dx, FunctionSpace, IntervalMesh, pi, TestFunction, TrialFunction
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory
from rbnics.eim.problems.time_dependent_eim_approximation import TimeDependentEIMApproximation
from rbnics.eim.reduction_methods.time_dependent_eim_approximation_reduction_method import (
    TimeDependentEIMApproximationReductionMethod)
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_10(expression_type, basis_generation):
    """
    This test is an extension of test 01.
    The aim of this script is to test the detection of time dependent parametrized expression.
    * EIM: test the case when the expression to be interpolated is time dependent.
    * DEIM: test interpolation of form with a time dependent integrand function.
    """

    class MockTimeDependentProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

            # Minimal subset of a time dependent ParametrizedDifferentialProblem
            self.t0 = 0.
            self.t = 0.
            self.dt = 0.
            self.T = 0.

        def name(self):
            return "MockTimeDependentProblem_10_" + expression_type + "_" + basis_generation

        def set_initial_time(self, t0):
            self.t0 = t0

        def set_time(self, t):
            self.t = t

        def set_time_step_size(self, dt):
            self.dt = dt

        def set_final_time(self, T):
            self.T = T

    class ParametrizedFunctionApproximation(TimeDependentEIMApproximation):
        def __init__(self, V, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mock_time_dependent_problem = MockTimeDependentProblem(V)
            f = ParametrizedExpression(
                mock_time_dependent_problem, "(1-x[0])*cos(3*pi*(1+t)*(1+x[0]))*exp(-(1+t)*(1+x[0]))", mu=(), t=0.,
                element=V.ufl_element())
            #
            folder_prefix = os.path.join("test_eim_approximation_10_tempdir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                TimeDependentEIMApproximation.__init__(
                    self, mock_time_dependent_problem, ParametrizedExpressionFactory(f), folder_prefix,
                    basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(V)
                form = f * v * dx
                # Call Parent constructor
                TimeDependentEIMApproximation.__init__(
                    self, mock_time_dependent_problem, ParametrizedTensorFactory(form), folder_prefix,
                    basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(V)
                v = TestFunction(V)
                form = f * u * v * dx
                # Call Parent constructor
                TimeDependentEIMApproximation.__init__(
                    self, mock_time_dependent_problem, ParametrizedTensorFactory(form), folder_prefix,
                    basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = IntervalMesh(100, -1., 1.)

    # 2. Create Finite Element space (Lagrange P1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # 3. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(V, expression_type, basis_generation)
    mu_range = []
    parametrized_function_approximation.set_mu_range(mu_range)
    parametrized_function_approximation.set_time_step_size(1.e-10)
    parametrized_function_approximation.set_final_time(pi - 1)

    # 4. Prepare reduction with EIM
    parametrized_function_reduction_method = TimeDependentEIMApproximationReductionMethod(
        parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(30)
    parametrized_function_reduction_method.set_tolerance(0.)

    # 5. Perform the offline phase
    parametrized_function_reduction_method.initialize_training_set(51, time_sampling=EquispacedDistribution())
    reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

    # 6. Perform an online solve
    online_mu = ()
    online_t = 0.
    reduced_parametrized_function_approximation.set_mu(online_mu)
    reduced_parametrized_function_approximation.set_time(online_t)
    reduced_parametrized_function_approximation.solve()

    # 7. Perform an error analysis
    parametrized_function_reduction_method.initialize_testing_set(100)
    parametrized_function_reduction_method.error_analysis()
