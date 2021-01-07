# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import (dx, grad, inner, Point, RectangleMesh, TensorElement, TestFunction, TrialFunction,
                    VectorFunctionSpace)
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_07(expression_type, basis_generation):
    """
    The aim of this test is to check the interpolation of tensor valued functions.
    * EIM: test interpolation of a scalar function
    * DEIM: test interpolation of form with integrand given by the inner product of a vector valued function
      and some derivative of a test/trial functions of a vector function space
    """

    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

        def name(self):
            return "MockProblem_07_" + expression_type + "_" + basis_generation

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, V, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mock_problem = MockProblem(V)
            f_expression = (
                ("1/sqrt(pow(x[0]-mu[0], 2) + pow(x[1]-mu[1], 2) + 0.01)",
                 "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )"),
                ("10.*(1-x[0])*cos(3*pi*(pi+mu[1])*(1+x[1]))*exp(-(pi+mu[0])*(1+x[0]))",
                 "sqrt(pow(x[0]-mu[0], 2) + pow(x[1]-mu[1], 2) + 0.01)")
            )
            tensor_element = TensorElement(V.sub(0).ufl_element())
            f = ParametrizedExpression(mock_problem, f_expression, mu=(-1., -1.), element=tensor_element)
            #
            folder_prefix = os.path.join("test_eim_approximation_07_tempdir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedExpressionFactory(f), folder_prefix, basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(V)
                form = inner(f, grad(v)) * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(V)
                v = TestFunction(V)
                form = inner(grad(u) * f, grad(v)) * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = RectangleMesh(Point(0.1, 0.1), Point(0.9, 0.9), 20, 20)

    # 2. Create Finite Element space (Lagrange P1)
    V = VectorFunctionSpace(mesh, "Lagrange", 1)

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
