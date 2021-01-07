# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import (div, dx, FiniteElement, FunctionSpace, grad, MixedElement, inner, split, Point, RectangleMesh,
                    TestFunction, TrialFunction, VectorElement)
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedTensorFactory
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_05(expression_type, basis_generation):
    """
    This test is combination of tests 01-03.
    The aim of this script is to test that integration correctly handles several parametrized expressions.
    * EIM: not applicable.
    * DEIM: several parametrized expressions are combined when defining forms.
    """

    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

        def name(self):
            return "MockProblem_05_" + expression_type + "_" + basis_generation

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, V, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mock_problem = MockProblem(V)
            f1 = ParametrizedExpression(
                mock_problem, "1/sqrt(pow(x[0]-mu[0], 2) + pow(x[1]-mu[1], 2) + 0.01)", mu=(-1., -1.),
                element=V.sub(1).ufl_element())
            f2 = ParametrizedExpression(
                mock_problem, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(-1., -1.),
                element=V.sub(1).ufl_element())
            f3 = ParametrizedExpression(
                mock_problem, "(1-x[0])*cos(3*pi*(pi+mu[1])*(1+x[1]))*exp(-(pi+mu[0])*(1+x[0]))", mu=(-1., -1.),
                element=V.sub(1).ufl_element())
            #
            folder_prefix = os.path.join("test_eim_approximation_05_tempdir", expression_type, basis_generation)
            assert expression_type in ("Vector", "Matrix")
            if expression_type == "Vector":
                vq = TestFunction(V)
                (v, q) = split(vq)
                form = f1 * v[0] * dx + f2 * v[1] * dx + f3 * q * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                up = TrialFunction(V)
                vq = TestFunction(V)
                (u, p) = split(up)
                (v, q) = split(vq)
                form = f1 * inner(grad(u), grad(v)) * dx + f2 * p * div(v) * dx + f3 * q * div(u) * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, mock_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = RectangleMesh(Point(0.1, 0.1), Point(0.9, 0.9), 20, 20)

    # 2. Create Finite Element space (Lagrange P1)
    element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    element = MixedElement(element_0, element_1)
    V = FunctionSpace(mesh, element)

    # 3. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(V, expression_type, basis_generation)
    mu_range = [(-1., -0.01), (-1., -0.01)]
    parametrized_function_approximation.set_mu_range(mu_range)

    # 4. Prepare reduction with EIM
    parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(20)
    parametrized_function_reduction_method.set_tolerance(0.)

    # 5. Perform the offline phase
    parametrized_function_reduction_method.initialize_training_set(100, sampling=EquispacedDistribution())
    reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

    # 6. Perform an online solve
    online_mu = (-1., -1.)
    reduced_parametrized_function_approximation.set_mu(online_mu)
    reduced_parametrized_function_approximation.solve()

    # 7. Perform an error analysis
    parametrized_function_reduction_method.initialize_testing_set(100)
    parametrized_function_reduction_method.error_analysis()
