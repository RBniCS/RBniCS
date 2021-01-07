# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import dx, FunctionSpace, Point, RectangleMesh, TestFunction, TrialFunction
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_02(expression_type, basis_generation):
    """
    This is a second basic test for EIM/DEIM, based on the test case of section 3.3.2 of
        S. Chaturantabut and D. C. Sorensen
        Nonlinear Model Reduction via Discrete Empirical Interpolation
        SIAM Journal on Scientific Computing 2010 32:5, 2737-2764
    * EIM: test interpolation of a scalar function
    * DEIM: test interpolation of form with scalar integrand on a scalar space
    """

    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

        def name(self):
            return "MockProblem_02_" + expression_type + "_" + basis_generation

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, V, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mock_problem = MockProblem(V)
            f = ParametrizedExpression(
                mock_problem, "1/sqrt(pow(x[0]-mu[0], 2) + pow(x[1]-mu[1], 2) + 0.01)", mu=(-1., -1.),
                element=V.ufl_element())
            #
            folder_prefix = os.path.join("test_eim_approximation_02_tempdir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedExpressionFactory(f), folder_prefix, basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(V)
                form = f * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(V)
                v = TestFunction(V)
                form = f * u * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = RectangleMesh(Point(0.1, 0.1), Point(0.9, 0.9), 20, 20)

    # 2. Create Finite Element space (Lagrange P1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # 3. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(V, expression_type, basis_generation)
    mu_range = [(-1., -0.01), (-1., -0.01)]
    parametrized_function_approximation.set_mu_range(mu_range)

    # 4. Prepare reduction with EIM
    parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(50)
    parametrized_function_reduction_method.set_tolerance(0.)

    # 5. Perform the offline phase
    parametrized_function_reduction_method.initialize_training_set(225, sampling=EquispacedDistribution())
    reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

    # 6. Perform an online solve
    online_mu = (-1., -1.)
    reduced_parametrized_function_approximation.set_mu(online_mu)
    reduced_parametrized_function_approximation.solve()

    # 7. Perform an error analysis
    parametrized_function_reduction_method.initialize_testing_set(225)
    parametrized_function_reduction_method.error_analysis()
