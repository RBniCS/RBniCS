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
from RBniCS import EquispacedDistribution
from RBniCS.backends import ParametrizedTensorFactory
from RBniCS.backends.fenics import ParametrizedExpressionFactory as ParametrizedExpressionFactory_Base
from RBniCS.backends.fenics.wrapping_utils import ParametrizedConstant
from RBniCS.eim.problems.eim_approximation import EIMApproximation
from RBniCS.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod

class ParametrizedExpressionFactory(ParametrizedExpressionFactory_Base):
    def __init__(self, expression, mesh):
        ParametrizedExpressionFactory_Base.__init__(self, expression)
        # Use a ridiculously high finite element space to have an accuracy comparable to the one of test 1,
        # where exact evaluation is carried out
        self._space = FunctionSpace(mesh, "CG", 10)
        
class ParametrizedFunctionApproximation(EIMApproximation):
    def __init__(self, V, expression_type, basis_generation):
        self.V = V
        # Parametrized function to be interpolated
        x = SpatialCoordinate(V.mesh())
        mu = ParametrizedConstant(self, "mu[0]", mu=(1., ))
        f = (1-x[0])*cos(3*pi*mu*(1+x[0]))*exp(-mu*(1+x[0]))
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedExpressionFactory(f, V.mesh()), "test_eim_approximation_9_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(V)
            form = f*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_9_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(V)
            v = TestFunction(V)
            form = f*u*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_9_matrix.output_dir", basis_generation)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid expression_type")

# 1. Create the mesh for this test
mesh = IntervalMesh(100, -1., 1.)

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the ParametrizedFunctionApproximation class
expression_type = "Function" # Function or Vector or Matrix
basis_generation = "Greedy" # Greedy or POD
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
