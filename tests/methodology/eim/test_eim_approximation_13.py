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
from RBniCS import EquispacedDistribution, ParametrizedExpression
from RBniCS.backends import ParametrizedExpressionFactory, ParametrizedTensorFactory
from RBniCS.problems.base import ParametrizedProblem
from RBniCS.eim.problems.time_dependent_eim_approximation import TimeDependentEIMApproximation
from RBniCS.eim.reduction_methods.time_dependent_eim_approximation_reduction_method import TimeDependentEIMApproximationReductionMethod

class MockTimeDependentProblem(ParametrizedProblem):
    def __init__(self, V, **kwargs):
        ParametrizedProblem.__init__(self, "")
        self.V = V
        
        # Minimal subset of a time dependent ParametrizedDifferentialProblem
        self.t = 0
        self.dt = 0
        self.T = 0
        self.set_mu_range([])
        
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
        self.mock_time_dependent_problem = MockTimeDependentProblem(V)
        f = ParametrizedExpression(self.mock_time_dependent_problem, "(1-x[0])*cos(3*pi*(1+t)*(1+x[0]))*exp(-(1+t)*(1+x[0]))", mu=(), t=0., element=V.ufl_element())
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            TimeDependentEIMApproximation.__init__(self, self.mock_time_dependent_problem, ParametrizedExpressionFactory(self.mock_time_dependent_problem, f), "test_eim_approximation_13_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(V)
            form = f*v*dx
            # Call Parent constructor
            TimeDependentEIMApproximation.__init__(self, self.mock_time_dependent_problem, ParametrizedTensorFactory(self.mock_time_dependent_problem, form), "test_eim_approximation_13_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(V)
            v = TestFunction(V)
            form = f*u*v*dx
            # Call Parent constructor
            TimeDependentEIMApproximation.__init__(self, self.mock_time_dependent_problem, ParametrizedTensorFactory(self.mock_time_dependent_problem, form), "test_eim_approximation_13_matrix.output_dir", basis_generation)
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
mu_range = []
parametrized_function_approximation.set_mu_range(mu_range)
parametrized_function_approximation.set_time_step_size(1.e-10)
parametrized_function_approximation.set_final_time(pi - 1)

# 4. Prepare reduction with EIM
parametrized_function_reduction_method = TimeDependentEIMApproximationReductionMethod(parametrized_function_approximation)
parametrized_function_reduction_method.set_Nmax(30)

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
