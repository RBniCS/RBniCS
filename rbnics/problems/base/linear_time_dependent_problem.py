# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.linear_problem import LinearProblem
from rbnics.problems.base.time_dependent_problem import TimeDependentProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearProblem, TimeDependentProblem)
def LinearTimeDependentProblem(ParametrizedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentProblem_Class(ParametrizedDifferentialProblem_DerivedClass):
        def __init__(self, V, **kwargs):
            # Call the parent initialization
            ParametrizedDifferentialProblem_DerivedClass.__init__(self, V, **kwargs)
            # Set the problem type in time stepping parameters
            self._time_stepping_parameters["problem_type"] = "linear"

    # return value (a class) for the decorator
    return LinearTimeDependentProblem_Class
