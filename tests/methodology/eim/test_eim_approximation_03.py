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
## @file 
#  @brief 
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import *
from rbnics import EquispacedDistribution, ParametrizedExpression
from rbnics.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod

class ParametrizedFunctionApproximation(EIMApproximation):
    def __init__(self, V, expression_type, basis_generation):
        self.V = V
        # Parametrized function to be interpolated
        f = ParametrizedExpression(self, "exp( - 2*pow(x[0]-mu[0], 2) - 2*pow(x[1]-mu[1], 2) )", mu=(0., 0.), element=V.ufl_element())
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedExpressionFactory(f), "test_eim_approximation_03_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(V)
            form = f*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_03_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(V)
            v = TestFunction(V)
            form = f*u*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_03_matrix.output_dir", basis_generation)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid expression_type")

# 1. Create the mesh for this test
mesh = Mesh("../../../tutorials/05_gaussian/data/gaussian.xml")

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the ParametrizedFunctionApproximation class
expression_type = "Function" # Function or Vector or Matrix
basis_generation = "Greedy" # Greedy or POD
parametrized_function_approximation = ParametrizedFunctionApproximation(V, expression_type, basis_generation)
mu_range = [(-1.0, 1.0), (-1.0, 1.0)]
parametrized_function_approximation.set_mu_range(mu_range)

# 4. Prepare reduction with EIM
parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
parametrized_function_reduction_method.set_Nmax(20)

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
