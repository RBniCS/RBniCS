# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.pod_galerkin_reduced_problem import PODGalerkinReducedProblem
from rbnics.problems.base.time_dependent_reduced_problem import TimeDependentReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(PODGalerkinReducedProblem, TimeDependentReducedProblem)
def TimeDependentPODGalerkinReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class TimeDependentPODGalerkinReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return TimeDependentPODGalerkinReducedProblem_Class
