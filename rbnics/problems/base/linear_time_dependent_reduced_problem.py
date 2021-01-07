# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.linear_reduced_problem import LinearReducedProblem
from rbnics.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearReducedProblem, TimeDependentReducedProblem)
def LinearTimeDependentReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        def __init__(self, truth_problem, **kwargs):
            # Call the parent initialization
            ParametrizedReducedDifferentialProblem_DerivedClass.__init__(self, truth_problem, **kwargs)
            # Set the problem type in time stepping parameters
            self._time_stepping_parameters["problem_type"] = "linear"

    # return value (a class) for the decorator
    return LinearTimeDependentReducedProblem_Class
