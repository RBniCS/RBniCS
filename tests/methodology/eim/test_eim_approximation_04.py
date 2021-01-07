# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import FunctionSpace, Measure, Mesh, MeshFunction, TestFunction, TrialFunction
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedTensorFactory
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem


@pytest.mark.parametrize("expression_type", ["Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_04(expression_type, basis_generation):
    """
    This test is an extension of test 03.
    The aim of this script is to test that integration of a subdomain is correctly handled by DEIM.
    * EIM: not applicable, as there is no difference with respect to test 03.
    * DEIM: the integral in form definitions is now restricted to a subdomain.
    """

    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            ParametrizedProblem.__init__(self, "")
            self.V = V

        def name(self):
            return "MockProblem_04_" + expression_type + "_" + basis_generation

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, V, subdomains, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mock_problem = MockProblem(V)
            f = ParametrizedExpression(
                mock_problem, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.),
                element=V.ufl_element())
            # Subdomain measure
            dx = Measure("dx")(subdomain_data=subdomains)(1)
            #
            folder_prefix = os.path.join("test_eim_approximation_04_tempdir", expression_type, basis_generation)
            assert expression_type in ("Vector", "Matrix")
            if expression_type == "Vector":
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
    mesh = Mesh("../../../tutorials/01_thermal_block/data/thermal_block.xml")
    subdomains = MeshFunction(
        "size_t", mesh, "../../../tutorials/01_thermal_block/data/thermal_block_physical_region.xml")

    # 2. Create Finite Element space (Lagrange P1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # 3. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(
        V, subdomains, expression_type, basis_generation)
    mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
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
