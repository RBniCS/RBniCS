# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.scm.problems.decorated_reduced_problem_with_stability_factor_evaluation import (
    DecoratedReducedProblemWithStabilityFactorEvaluation)
from rbnics.scm.problems.exact_stability_factor import ExactStabilityFactor
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor


@ReducedProblemDecoratorFor(ExactStabilityFactor)
def ExactStabilityFactorDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @DecoratedReducedProblemWithStabilityFactorEvaluation
    @PreserveClassName
    class ExactStabilityFactorDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return ExactStabilityFactorDecoratedReducedProblem_Class
