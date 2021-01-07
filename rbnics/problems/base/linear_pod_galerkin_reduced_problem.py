# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.linear_reduced_problem import LinearReducedProblem
from rbnics.problems.base.pod_galerkin_reduced_problem import PODGalerkinReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearReducedProblem, PODGalerkinReducedProblem)
def LinearPODGalerkinReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class LinearPODGalerkinReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearPODGalerkinReducedProblem_Class
