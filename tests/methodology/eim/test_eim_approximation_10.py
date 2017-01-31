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
from RBniCS.backends import BasisFunctionsMatrix, ParametrizedTensorFactory
from RBniCS.backends.fenics import ParametrizedExpressionFactory as ParametrizedExpressionFactory_Base
from RBniCS.backends.fenics.wrapping_utils import ParametrizedConstant
from RBniCS.backends.online import OnlineFunction
from RBniCS.eim.problems.eim_approximation import EIMApproximation
from RBniCS.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from RBniCS.problems.base import ParametrizedProblem
from RBniCS.reduction_methods.base import ReductionMethod
from RBniCS.utils.decorators import StoreMapFromProblemNameToProblem, StoreMapFromProblemToReducedProblem, StoreMapFromSolutionToProblem, sync_setters

class ParametrizedExpressionFactory(ParametrizedExpressionFactory_Base):
    def __init__(self, expression, mesh):
        ParametrizedExpressionFactory_Base.__init__(self, expression)
        # Use a ridiculously high finite element space to have an accuracy comparable to the one of test 1,
        # where exact evaluation is carried out
        self._space = FunctionSpace(mesh, "CG", 10)

@StoreMapFromProblemNameToProblem
@StoreMapFromSolutionToProblem
class MockProblem(ParametrizedProblem):
    def __init__(self, V, **kwargs):
        # Minimal subset of a ParametrizedDifferentialProblem
        self.V = V
        self._solution = Function(V)
        # Parametrized function to be interpolated
        x = SpatialCoordinate(V.mesh())
        mu = ParametrizedConstant(self, "mu[0]", mu=(1., ))
        self.f = (1-x[0])*cos(3*pi*mu*(1+x[0]))*exp(-mu*(1+x[0]))

class MockReductionMethod(ReductionMethod):
    def __init__(self, truth_problem, **kwargs):
        self.truth_problem = truth_problem
        self.reduced_problem = None
        
    def offline(self):
        self.reduced_problem = MockReducedProblem(self.truth_problem)
        return self.reduced_problem
        
    def error_analysis(self, N=None, **kwargs):
        pass
        
    def speedup_analysis(self, N=None, **kwargs):
        pass

@StoreMapFromProblemToReducedProblem
class MockReducedProblem(ParametrizedProblem):
    @sync_setters("truth_problem", "set_mu", "mu")
    @sync_setters("truth_problem", "set_mu_range", "mu_range")
    def __init__(self, truth_problem, **kwargs):
        # Minimal subset of a ParametrizedReducedDifferentialProblem
        self.truth_problem = truth_problem
        self.Z = BasisFunctionsMatrix(V)
        self.Z.init(["f"])
        self._solution = OnlineFunction(1)
        
    def solve(self):
        print "Solving mock reduced problem at mu =", self.mu
        # This is not really a reduced problem, it carries out the
        # exact interpolation by updating the basis functions matrix in
        # the truth problem
        self.Z.clear()
        f = project(self.truth_problem.f, self.truth_problem.V)
        self.Z.enrich(f)
        # Return the reduced solution
        self._solution.vector()[0] = 1.
        return self._solution
        
class ParametrizedFunctionApproximation(EIMApproximation):
    def __init__(self, truth_problem, expression_type, basis_generation):
        self.truth_problem = truth_problem
        self.V = truth_problem.V
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            EIMApproximation.__init__(self, truth_problem, ParametrizedExpressionFactory(self.truth_problem._solution, self.V.mesh()), "test_eim_approximation_10_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(self.V)
            form = self.truth_problem._solution*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, truth_problem, ParametrizedTensorFactory(form), "test_eim_approximation_10_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            form = self.truth_problem._solution*u*v*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, truth_problem, ParametrizedTensorFactory(form), "test_eim_approximation_10_matrix.output_dir", basis_generation)
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

# 4. Create a reduction method and run the offline phase to generate the corresponding
#    reduced problem
reduction_method = MockReductionMethod(problem)
reduced_problem = reduction_method.offline()

# 5. Allocate an object of the ParametrizedFunctionApproximation class
expression_type = "Vector" # Function or Vector or Matrix
basis_generation = "Greedy" # Greedy or POD
parametrized_function_approximation = ParametrizedFunctionApproximation(problem, expression_type, basis_generation)
parametrized_function_approximation.set_mu_range(mu_range)

# 6. Prepare reduction with EIM
parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
parametrized_function_reduction_method.set_Nmax(30)

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
