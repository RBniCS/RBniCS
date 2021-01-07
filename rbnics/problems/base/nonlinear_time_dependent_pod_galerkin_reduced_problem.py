# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.nonlinear_pod_galerkin_reduced_problem import NonlinearPODGalerkinReducedProblem
from rbnics.problems.base.time_dependent_pod_galerkin_reduced_problem import TimeDependentPODGalerkinReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearPODGalerkinReducedProblem, TimeDependentPODGalerkinReducedProblem)
def NonlinearTimeDependentPODGalerkinReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class NonlinearTimeDependentPODGalerkinReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearTimeDependentPODGalerkinReducedProblem_Class
