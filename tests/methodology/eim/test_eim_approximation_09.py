# Copyright (C) 2015-2017 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

import os
import pytest
from dolfin import cos, dx, exp, Expression, FunctionSpace, IntervalMesh, pi, project, SpatialCoordinate, TestFunction, TrialFunction
from rbnics import EquispacedDistribution
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory, SymbolicParameters
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod

@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_09(expression_type, basis_generation):
    """
    This test is an extension of test 01.
    The aim of this script is to test the detection of parametrized expression defined using SymbolicParameters for mu and
    SpatialCoordinates fo x.
    * EIM: test the case when the expression to be interpolated is an Operator (rather than an Expression).
    * DEIM: test interpolation of form with integrand function of type Operator (rather than Expression).
    """

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, V, expression_type, basis_generation):
            self.V = V
            # Parametrized function to be interpolated
            mu = SymbolicParameters(self, V, mu=(1., ))
            if expression_type == "Function":
                x = project(Expression("x[0]", element=V.ufl_element()), V) # SpatialCoordinate is not supported by FEniCS dP measure
                x = (x, )
            else:
                x = SpatialCoordinate(V.mesh())
            f = (1-x[0])*cos(3*pi*mu[0]*(1+x[0]))*exp(-mu[0]*(1+x[0]))
            #
            folder_prefix = os.path.join("test_eim_approximation_09.output_dir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                EIMApproximation.__init__(self, None, ParametrizedExpressionFactory(f), folder_prefix, basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(V)
                form = f*v*dx
                # Call Parent constructor
                EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(V)
                v = TestFunction(V)
                form = f*u*v*dx
                # Call Parent constructor
                EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else: # impossible to arrive here anyway thanks to the assert
                raise AssertionError("Invalid expression_type")

    # 1. Create the mesh for this test
    mesh = IntervalMesh(100, -1., 1.)

    # 2. Create Finite Element space (Lagrange P1)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # 3. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(V, expression_type, basis_generation)
    mu_range = [(1., pi), ]
    parametrized_function_approximation.set_mu_range(mu_range)

    # 4. Prepare reduction with EIM
    parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(30)

    # 5. Perform the offline phase
    parametrized_function_reduction_method.initialize_training_set(51, sampling=EquispacedDistribution())
    reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

    # 6. Perform an online solve
    online_mu = (1., )
    reduced_parametrized_function_approximation.set_mu(online_mu)
    reduced_parametrized_function_approximation.solve()

    # 7. Perform an error analysis
    parametrized_function_reduction_method.initialize_testing_set(100)
    parametrized_function_reduction_method.error_analysis()
