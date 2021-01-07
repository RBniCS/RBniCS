# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.scm.problems.decorated_reduced_problem_with_stability_factor_evaluation import (
    DecoratedReducedProblemWithStabilityFactorEvaluation)
from rbnics.scm.problems.scm import SCM
from rbnics.utils.decorators import PreserveClassName, ReducedProblemDecoratorFor


@ReducedProblemDecoratorFor(SCM)
def SCMDecoratedReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @DecoratedReducedProblemWithStabilityFactorEvaluation
    @PreserveClassName
    class SCMDecoratedReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return SCMDecoratedReducedProblem_Class
