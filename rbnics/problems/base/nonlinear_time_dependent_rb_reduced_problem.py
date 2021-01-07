# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.nonlinear_rb_reduced_problem import NonlinearRBReducedProblem
from rbnics.problems.base.time_dependent_rb_reduced_problem import TimeDependentRBReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearRBReducedProblem, TimeDependentRBReducedProblem)
def NonlinearTimeDependentRBReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class NonlinearTimeDependentRBReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearTimeDependentRBReducedProblem_Class
