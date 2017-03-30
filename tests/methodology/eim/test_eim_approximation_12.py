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

from __future__ import print_function
from dolfin import *
from rbnics import EquispacedDistribution
from rbnics.backends import BasisFunctionsMatrix, GramSchmidt, ParametrizedExpressionFactory, ParametrizedTensorFactory, transpose
from rbnics.backends.fenics.wrapping import ParametrizedConstant
from rbnics.backends.online import OnlineFunction
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.decorators import StoreMapFromProblemNameToProblem, StoreMapFromProblemToTrainingStatus, StoreMapFromSolutionToProblem
from rbnics.utils.mpi import print

@StoreMapFromProblemNameToProblem
@StoreMapFromProblemToTrainingStatus
@StoreMapFromSolutionToProblem
class MockProblem(ParametrizedProblem):
    def __init__(self, V, **kwargs):
        # Call parent
        ParametrizedProblem.__init__(self, "test_eim_approximation_12_mock_problem.output_dir")
        # Minimal subset of a ParametrizedDifferentialProblem
        self.V = V
        self._solution = Function(V)
        self.components = ["f"]
        # Parametrized function to be interpolated
        x = SpatialCoordinate(V.mesh())
        mu = ParametrizedConstant(self, "mu[0]", mu=(1., ))
        self.f = (1-x[0])*cos(3*pi*mu*(1+x[0]))*exp(-mu*(1+x[0]))
        # Inner product
        f = TrialFunction(self.V)
        g = TestFunction(self.V)
        self.X = assemble(f*g*dx)
        
    def init(self):
        pass
        
    def solve(self):
        print("solving mock problem at mu =", self.mu)
        assert not hasattr(self, "_is_solving")
        self._is_solving = True
        project(self.f, self.V, function=self._solution)
        delattr(self, "_is_solving")
        return self._solution
        
class ParametrizedFunctionApproximation(EIMApproximation):
    def __init__(self, truth_problem, expression_type, basis_generation, function):
        self.V = truth_problem.V
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedExpressionFactory(truth_problem._solution), "test_eim_approximation_12_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(self.V)
            form = truth_problem._solution*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_12_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            form = truth_problem._solution*u*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_12_matrix.output_dir", basis_generation)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid expression_type")

# 1. Create the mesh for this test
mesh = IntervalMesh(100, -1., 1.)

# 2. Create Finite Element space (Lagrange P1)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Create a parametrized problem
problem = MockProblem(V)
mu_range = [(1., pi), ]
problem.set_mu_range(mu_range)

# 4. Postpone generation of the reduced problem

# 5. Allocate an object of the ParametrizedFunctionApproximation class
expression_type = "Function" # Function or Vector or Matrix
basis_generation = "Greedy" # Greedy or POD
function = lambda u: exp(u)
parametrized_function_approximation = ParametrizedFunctionApproximation(problem, expression_type, basis_generation, function)
parametrized_function_approximation.set_mu_range(mu_range)

# 6. Prepare reduction with EIM
parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
parametrized_function_reduction_method.set_Nmax(12)

# 7. Perform EIM offline phase
parametrized_function_reduction_method.initialize_training_set(51, sampling=EquispacedDistribution())
reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

# 8. Perform EIM online solve
online_mu = (1., )
reduced_parametrized_function_approximation.set_mu(online_mu)
reduced_parametrized_function_approximation.solve()

# 9. Perform EIM error analysis
parametrized_function_reduction_method.initialize_testing_set(100)
parametrized_function_reduction_method.error_analysis()
