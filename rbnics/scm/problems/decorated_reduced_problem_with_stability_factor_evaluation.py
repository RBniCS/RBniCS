# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from abc import abstractmethod
from math import sqrt
from numpy import isclose
from rbnics.scm.problems.parametrized_stability_factor_reduced_eigenproblem import (
    ParametrizedStabilityFactorReducedEigenProblem)
from rbnics.utils.decorators import PreserveClassName


def DecoratedReducedProblemWithStabilityFactorEvaluation(ParametrizedReducedDifferentialProblem_DerivedClass):
    from rbnics.problems.elliptic import EllipticRBReducedProblem, EllipticCoerciveRBReducedProblem
    from rbnics.problems.stokes import StokesRBReducedProblem

    @PreserveClassName
    class DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base(
            ParametrizedReducedDifferentialProblem_DerivedClass):
        # Default initialization of members
        def __init__(self, truth_problem, **kwargs):
            # Eigen solver parameters (to be initialized before calling parent initialization as it may
            # update them).
            self._eigen_solver_parameters = {
                "stability_factor": dict()
            }
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Additional basis functions matrix required by stability factor computations
            self.stability_factor_basis_functions = None  # BasisFunctionsMatrix
            # Stability factor eigen problem
            self.stability_factor_calculator = ParametrizedStabilityFactorReducedEigenProblem(
                self, "smallest", self._eigen_solver_parameters["stability_factor"],
                os.path.join(truth_problem.name(), "reduced_stability_factor__NEVER_USED"))

        def init(self, current_stage="online"):
            # Call parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.init(self, current_stage)
            # Initialize attributes related to stability factor computations as well
            self._init_stability_factor_basis_functions(current_stage)
            self._init_stability_factor_calculator(current_stage)

        @abstractmethod
        def _init_stability_factor_basis_functions(self, current_stage="online"):
            pass

        def _init_stability_factor_calculator(self, current_stage="online"):
            self.stability_factor_calculator.init(current_stage)

        # Return the lower bound for the stability factor.
        def get_stability_factor_lower_bound(self, N=None, **kwargs):
            # Call the exact evaluation, since its computational cost is low because we are dealing with
            # left-hand and right-hand side matrices of small dimensions
            return self.evaluate_stability_factor(N, **kwargs)

        def evaluate_stability_factor(self, N=None, **kwargs):
            (minimum_eigenvalue, _) = self.stability_factor_calculator.solve(N, **kwargs)
            return minimum_eigenvalue

    # Elliptic coercive RB reduced problem specialization
    if issubclass(ParametrizedReducedDifferentialProblem_DerivedClass, EllipticCoerciveRBReducedProblem):

        @PreserveClassName
        class DecoratedReducedProblemWithStabilityFactorEvaluation_Class(
                DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base):
            def __init__(self, truth_problem, **kwargs):
                # Call the parent initialization
                DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base.__init__(self, truth_problem, **kwargs)
                # Update eigen solver parameters
                self._eigen_solver_parameters["stability_factor"]["problem_type"] = "gen_hermitian"

            def _init_stability_factor_basis_functions(self, current_stage="online"):
                self.stability_factor_basis_functions = self.basis_functions

    # Elliptic non-coercive (needs to be after the coercive case) or Stokes RB reduced problem specialization
    elif issubclass(ParametrizedReducedDifferentialProblem_DerivedClass,
                    (EllipticRBReducedProblem, StokesRBReducedProblem)):

        @PreserveClassName
        class DecoratedReducedProblemWithStabilityFactorEvaluation_Class(
                DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base):
            def __init__(self, truth_problem, **kwargs):
                # Call the parent initialization
                DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base.__init__(self, truth_problem, **kwargs)
                # Update eigen solver parameters
                self._eigen_solver_parameters["stability_factor"]["problem_type"] = "gen_non_hermitian"

            def _init_stability_factor_basis_functions(self, current_stage="online"):
                self.stability_factor_basis_functions = self.basis_functions  # TODO

            def evaluate_stability_factor(self, N=None, **kwargs):
                beta_squared = (
                    DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base.evaluate_stability_factor(
                        self, N, **kwargs))
                assert beta_squared > 0. or isclose(beta_squared, 0.)
                return sqrt(abs(beta_squared))

    # Unhandled case: will return an error due to abstract methods
    else:
        DecoratedReducedProblemWithStabilityFactorEvaluation_Class = (
            DecoratedReducedProblemWithStabilityFactorEvaluation_Class_Base)

    # return value (a class) for the decorator
    return DecoratedReducedProblemWithStabilityFactorEvaluation_Class
