# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import os
from rbnics.utils.decorators import PreserveClassName, ProblemDecoratorFor
from rbnics.scm.problems.parametrized_stability_factor_eigenproblem import ParametrizedStabilityFactorEigenProblem
from rbnics.scm.problems.decorated_problem_with_stability_factor_evaluation import (
    DecoratedProblemWithStabilityFactorEvaluation)


def ExactStabilityFactorDecoratedProblem(
    **decorator_kwargs
):
    from rbnics.scm.problems.exact_stability_factor import ExactStabilityFactor

    @ProblemDecoratorFor(ExactStabilityFactor)
    def ExactStabilityFactorDecoratedProblem_Decorator(ParametrizedDifferentialProblem_DerivedClass):

        @DecoratedProblemWithStabilityFactorEvaluation
        @PreserveClassName
        class ExactStabilityFactorDecoratedProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
            # Default initialization of members
            def __init__(self, V, **kwargs):
                # Eigen solver parameters (to be initialized before calling parent initialization as it may
                # update them).
                self._eigen_solver_parameters = {
                    "stability_factor": dict()
                }
                # Additional function space required by stability factor computations (to be initialized before
                # calling parent initialization as it will update them)
                self.stability_factor_V = None
                # Call the parent initialization
                ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
                # Additional terms required by stability factor computations
                self.terms.extend(["stability_factor_left_hand_matrix", "stability_factor_right_hand_matrix"])
                self.terms_order.update({"stability_factor_left_hand_matrix": 2,
                                         "stability_factor_right_hand_matrix": 2})
                # Stability factor eigen problem
                self.stability_factor_calculator = ParametrizedStabilityFactorEigenProblem(
                    self, "smallest", self._eigen_solver_parameters["stability_factor"],
                    os.path.join(self.name(), "exact_stability_factor"))
                self.stability_factor_lower_bound_calculator = None

            # Initialize data structures required for the online phase
            def init(self):
                # Call to Parent
                ParametrizedDifferentialProblem_DerivedClass.init(self)
                # Init exact stability factor computations
                self.stability_factor_calculator.init()

        # return value (a class) for the decorator
        return ExactStabilityFactorDecoratedProblem_Class

    # return the decorator itself
    return ExactStabilityFactorDecoratedProblem_Decorator
