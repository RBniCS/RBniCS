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

from __future__ import print_function
from dolfin import *
from rbnics import EquispacedDistribution, ExactParametrizedFunctions
from rbnics.backends import BasisFunctionsMatrix, GramSchmidt, ParametrizedExpressionFactory, ParametrizedTensorFactory, SymbolicParameters, transpose
from rbnics.backends.online import OnlineFunction
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem
from rbnics.utils.decorators import StoreMapFromProblemNameToProblem, StoreMapFromProblemToTrainingStatus, StoreMapFromSolutionToProblem
from rbnics.utils.mpi import print

"""
This test is the version of test 21 where high fidelity solution is used in place of reduced order one.
"""

@ExactParametrizedFunctions("offline")
@StoreMapFromProblemNameToProblem
@StoreMapFromProblemToTrainingStatus
@StoreMapFromSolutionToProblem
class MockProblem(ParametrizedProblem):
    def __init__(self, V, **kwargs):
        # Call parent
        ParametrizedProblem.__init__(self, "test_eim_approximation_22_mock_problem.output_dir")
        # Minimal subset of a ParametrizedDifferentialProblem
        self.V = V
        self._solution = Function(V)
        self.components = ["f"]
        # Parametrized function to be interpolated
        x = SpatialCoordinate(V.mesh())
        mu = SymbolicParameters(self, V, mu=(-1., -1.))
        self.f00 = 1./sqrt(pow(x[0]-mu[0], 2) + pow(x[1]-mu[1], 2) + 0.01)
        self.f01 = 1./sqrt(pow(x[0]-mu[0], 4) + pow(x[1]-mu[1], 4) + 0.01)
        # Inner product
        f = TrialFunction(self.V)
        g = TestFunction(self.V)
        self.X = assemble(inner(f, g)*dx)
        # Collapsed vector and space
        self.V0 = V.sub(0).collapse()
        self.V00 = V.sub(0).sub(0).collapse()
        self.V1 = V.sub(1).collapse()
        
    def name(self):
        return "MockProblem"
        
    def init(self):
        pass
        
    def solve(self):
        print("solving mock problem at mu =", self.mu)
        assert not hasattr(self, "_is_solving")
        self._is_solving = True
        f00 = project(self.f00, self.V00)
        f01 = project(self.f01, self.V00)
        assign(self._solution.sub(0).sub(0), f00)
        assign(self._solution.sub(0).sub(1), f01)
        delattr(self, "_is_solving")
        return self._solution
        
class ParametrizedFunctionApproximation(EIMApproximation):
    def __init__(self, truth_problem, expression_type, basis_generation):
        self.V = truth_problem.V0
        (f0, _) = split(truth_problem._solution)
        #
        assert expression_type in ("Function", "Vector", "Matrix")
        if expression_type == "Function":
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedExpressionFactory(f0.dx(0)), "test_eim_approximation_22_function.output_dir", basis_generation)
        elif expression_type == "Vector":
            v = TestFunction(self.V)
            form = inner(f0.dx(0), v.dx(0))*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_22_vector.output_dir", basis_generation)
        elif expression_type == "Matrix":
            u = TrialFunction(self.V)
            v = TestFunction(self.V)
            form = inner(f0.dx(0), u.dx(0))*v[0]*dx
            # Call Parent constructor
            EIMApproximation.__init__(self, None, ParametrizedTensorFactory(form), "test_eim_approximation_22_matrix.output_dir", basis_generation)
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid expression_type")

# 1. Create the mesh for this test
mesh = RectangleMesh(Point(0.1, 0.1), Point(0.9, 0.9), 20, 20)

# 2. Create Finite Element space (Lagrange P1)
element_0 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
element_1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
element   = MixedElement(element_0, element_1)
V = FunctionSpace(mesh, element)

# 3. Create a parametrized problem
problem = MockProblem(V)
mu_range = [(-1., -0.01), (-1., -0.01)]
problem.set_mu_range(mu_range)

# 4. Postpone generation of the reduced problem

# 5. Allocate an object of the ParametrizedFunctionApproximation class
expression_type = "Function" # Function or Vector or Matrix
basis_generation = "Greedy" # Greedy or POD
parametrized_function_approximation = ParametrizedFunctionApproximation(problem, expression_type, basis_generation)
parametrized_function_approximation.set_mu_range(mu_range)

# 6. Prepare reduction with EIM
parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
parametrized_function_reduction_method.set_Nmax(16)

# 7. Perform EIM offline phase
parametrized_function_reduction_method.initialize_training_set(64, sampling=EquispacedDistribution())
reduced_parametrized_function_approximation = parametrized_function_reduction_method.offline()

# 8. Perform EIM online solve
online_mu = (-1., -1.)
reduced_parametrized_function_approximation.set_mu(online_mu)
reduced_parametrized_function_approximation.solve()

# 9. Perform EIM error analysis
parametrized_function_reduction_method.initialize_testing_set(100)
parametrized_function_reduction_method.error_analysis()
