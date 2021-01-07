# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.linear_rb_reduced_problem import LinearRBReducedProblem
from rbnics.problems.base.time_dependent_rb_reduced_problem import TimeDependentRBReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearRBReducedProblem, TimeDependentRBReducedProblem)
def LinearTimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearTimeDependentRBReducedProblem_Class
