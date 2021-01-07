# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.linear_pod_galerkin_reduced_problem import LinearPODGalerkinReducedProblem
from rbnics.problems.base.time_dependent_pod_galerkin_reduced_problem import TimeDependentPODGalerkinReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearPODGalerkinReducedProblem, TimeDependentPODGalerkinReducedProblem)
def LinearTimeDependentPODGalerkinReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentPODGalerkinReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearTimeDependentPODGalerkinReducedProblem_Class
