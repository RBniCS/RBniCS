# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import cos, dx, exp, FunctionSpace, IntervalMesh, pi, SpatialCoordinate, TestFunction, TrialFunction
from rbnics import EquispacedDistribution
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory, SymbolicParameters
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_09(expression_type, basis_generation):
    """
    This test is an extension of test 01.
    The aim of this script is to test the detection of parametrized expression defined using SymbolicParameters
    for mu and SpatialCoordinates for x.
    * EIM: test the case when the expression to be interpolated is an Operator (rather than an Expression).
    * DEIM: test interpolation of form with integrand function of type Operator (rather than Expression).
    """

    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

        def name(self):
            return "MockProblem_09_" + expression_type + "_" + basis_generation

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, mock_problem, expression_type, basis_generation):
            self.V = mock_problem.V
            # Parametrized function to be interpolated
            mu = SymbolicParameters(mock_problem, self.V, (1., ))
            x = SpatialCoordinate(self.V.mesh())
            f = (1 - x[0]) * cos(3 * pi * mu[0] * (1 + x[0])) * exp(- mu[0] * (1 + x[0]))
            #
            folder_prefix = os.path.join("test_eim_approximation_09_tempdir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedExpressionFactory(f), folder_prefix, basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(self.V)
                form = f * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(self.V)
                v = TestFunction(self.V)
                form = f * u * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = IntervalMesh(100, -1., 1.)

    # 2. Create Finite Element space (Lagrange P1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # 3. Create a parametrized problem
    mock_problem = MockProblem(V)
    mu_range = [(1., pi), ]
    mock_problem.set_mu_range(mu_range)

    # 4. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(
        mock_problem, expression_type, basis_generation)

    # 5. Prepare reduction with EIM
    parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(30)
    parametrized_function_reduction_method.set_tolerance(0.)

    # 6. Perform the offline phase
    parametrized_function_reduction_method.initialize_training_set(51, sampling=EquispacedDistribution())
    reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

    # 7. Perform an online solve
    online_mu = (1., )
    reduced_parametrized_function_approximation.set_mu(online_mu)
    reduced_parametrized_function_approximation.solve()

    # 8. Perform an error analysis
    parametrized_function_reduction_method.initialize_testing_set(100)
    parametrized_function_reduction_method.error_analysis()
