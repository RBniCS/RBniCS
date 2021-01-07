# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
import pytest
from dolfin import (assemble, cos, dx, exp, Function, FunctionSpace, IntervalMesh, pi, project, SpatialCoordinate,
                    TestFunction, TrialFunction)
from rbnics import EquispacedDistribution
from rbnics.backends import (BasisFunctionsMatrix, GramSchmidt, ParametrizedExpressionFactory,
                             ParametrizedTensorFactory, SymbolicParameters, transpose)
from rbnics.backends.online import OnlineFunction
from rbnics.eim.problems.eim_approximation import EIMApproximation
from rbnics.eim.reduction_methods.eim_approximation_reduction_method import EIMApproximationReductionMethod
from rbnics.problems.base import ParametrizedProblem
from rbnics.reduction_methods.base import ReductionMethod
from rbnics.utils.decorators import (StoreMapFromProblemNameToProblem, StoreMapFromProblemToReducedProblem,
                                     StoreMapFromProblemToReductionMethod, StoreMapFromProblemToTrainingStatus,
                                     StoreMapFromSolutionToProblem, sync_setters, UpdateMapFromProblemToTrainingStatus)


@pytest.mark.parametrize("expression_type", ["Function", "Vector", "Matrix"])
@pytest.mark.parametrize("basis_generation", ["Greedy", "POD"])
def test_eim_approximation_11(expression_type, basis_generation):
    """
    The aim of this script is to test EIM/DEIM for nonlinear problems. A parametrized problem is defined
    and reduced by means of a reduction method. Then:
    * EIM: the expression to be interpolated is the solution of the nonlinear reduced problem.
    * DEIM: the form to be interpolated contains the solution of the nonlinear reduced problem.
    """

    @StoreMapFromProblemNameToProblem
    @StoreMapFromProblemToTrainingStatus
    @StoreMapFromSolutionToProblem
    class MockProblem(ParametrizedProblem):
        def __init__(self, V, **kwargs):
            # Call parent
            ParametrizedProblem.__init__(self, os.path.join(
                "test_eim_approximation_11_tempdir", expression_type, basis_generation, "mock_problem"))
            # Minimal subset of a ParametrizedDifferentialProblem
            self.V = V
            self._solution = Function(V)
            self.components = ["u"]
            # Parametrized function to be interpolated
            x = SpatialCoordinate(V.mesh())
            mu = SymbolicParameters(self, V, (1., ))
            self.f = (1 - x[0]) * cos(3 * pi * mu[0] * (1 + x[0])) * exp(- mu[0] * (1 + x[0]))
            # Inner product
            f = TrialFunction(self.V)
            g = TestFunction(self.V)
            self.inner_product = assemble(f * g * dx)

        def name(self):
            return "MockProblem_11_" + expression_type + "_" + basis_generation

        def init(self):
            pass

        def solve(self):
            assert not hasattr(self, "_is_solving")
            self._is_solving = True
            project(self.f, self.V, function=self._solution)
            delattr(self, "_is_solving")
            return self._solution

    @StoreMapFromProblemToReductionMethod
    @UpdateMapFromProblemToTrainingStatus
    class MockReductionMethod(ReductionMethod):
        def __init__(self, truth_problem, **kwargs):
            # Call parent
            ReductionMethod.__init__(self, os.path.join(
                "test_eim_approximation_11_tempdir", expression_type, basis_generation, "mock_problem"))
            # Minimal subset of a DifferentialProblemReductionMethod
            self.truth_problem = truth_problem
            self.reduced_problem = None
            # I/O
            self.folder["basis"] = os.path.join(self.truth_problem.folder_prefix, "basis")
            # Gram Schmidt
            self.GS = GramSchmidt(self.truth_problem.V, self.truth_problem.inner_product)

        def initialize_training_set(self, ntrain, enable_import=True, sampling=None, **kwargs):
            return ReductionMethod.initialize_training_set(
                self, self.truth_problem.mu_range, ntrain, enable_import, sampling, **kwargs)

        def initialize_testing_set(self, ntest, enable_import=False, sampling=None, **kwargs):
            return ReductionMethod.initialize_testing_set(
                self, self.truth_problem.mu_range, ntest, enable_import, sampling, **kwargs)

        def offline(self):
            self.reduced_problem = MockReducedProblem(self.truth_problem)
            if self.folder["basis"].create():  # basis folder was not available yet
                for mu in self.training_set:
                    self.truth_problem.set_mu(mu)
                    print("solving mock problem at mu =", self.truth_problem.mu)
                    f = self.truth_problem.solve()
                    self.update_basis_matrix(f)
                self.reduced_problem.basis_functions.save(self.folder["basis"], "basis")
            else:
                self.reduced_problem.basis_functions.load(self.folder["basis"], "basis")
            self._finalize_offline()
            return self.reduced_problem

        def update_basis_matrix(self, snapshot):
            new_basis_function = self.GS.apply(snapshot, self.reduced_problem.basis_functions)
            self.reduced_problem.basis_functions.enrich(new_basis_function)

        def error_analysis(self, N=None, **kwargs):
            pass

        def speedup_analysis(self, N=None, **kwargs):
            pass

    @StoreMapFromProblemToReducedProblem
    class MockReducedProblem(ParametrizedProblem):
        @sync_setters("truth_problem", "set_mu", "mu")
        @sync_setters("truth_problem", "set_mu_range", "mu_range")
        def __init__(self, truth_problem, **kwargs):
            # Call parent
            ParametrizedProblem.__init__(self, os.path.join(
                "test_eim_approximation_11_tempdir", expression_type, basis_generation, "mock_problem"))
            # Minimal subset of a ParametrizedReducedDifferentialProblem
            self.truth_problem = truth_problem
            self.basis_functions = BasisFunctionsMatrix(self.truth_problem.V)
            self.basis_functions.init(self.truth_problem.components)
            self._solution = None

        def solve(self):
            print("solving mock reduced problem at mu =", self.mu)
            assert not hasattr(self, "_is_solving")
            self._is_solving = True
            f = self.truth_problem.solve()
            f_N = transpose(self.basis_functions) * self.truth_problem.inner_product * f
            # Return the reduced solution
            self._solution = OnlineFunction(f_N)
            delattr(self, "_is_solving")
            return self._solution

    class ParametrizedFunctionApproximation(EIMApproximation):
        def __init__(self, truth_problem, expression_type, basis_generation, function):
            self.V = truth_problem.V
            #
            folder_prefix = os.path.join("test_eim_approximation_11_tempdir", expression_type, basis_generation)
            assert expression_type in ("Function", "Vector", "Matrix")
            if expression_type == "Function":
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, truth_problem, ParametrizedExpressionFactory(truth_problem._solution), folder_prefix,
                    basis_generation)
            elif expression_type == "Vector":
                v = TestFunction(self.V)
                form = truth_problem._solution * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, truth_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            elif expression_type == "Matrix":
                u = TrialFunction(self.V)
                v = TestFunction(self.V)
                form = truth_problem._solution * u * v * dx
                # Call Parent constructor
                EIMApproximation.__init__(
                    self, truth_problem, ParametrizedTensorFactory(form), folder_prefix, basis_generation)
            else:  # impossible to arrive here anyway thanks to the assert
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
    reduction_method.initialize_training_set(12, sampling=EquispacedDistribution())
    reduction_method.offline()

    # 5. Allocate an object of the ParametrizedFunctionApproximation class
    parametrized_function_approximation = ParametrizedFunctionApproximation(
        problem, expression_type, basis_generation, lambda u: exp(u))
    parametrized_function_approximation.set_mu_range(mu_range)

    # 6. Prepare reduction with EIM
    parametrized_function_reduction_method = EIMApproximationReductionMethod(parametrized_function_approximation)
    parametrized_function_reduction_method.set_Nmax(12)
    parametrized_function_reduction_method.set_tolerance(0.)

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
