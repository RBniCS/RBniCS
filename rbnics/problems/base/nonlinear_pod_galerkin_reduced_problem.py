# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base.nonlinear_reduced_problem import NonlinearReducedProblem
from rbnics.problems.base.pod_galerkin_reduced_problem import PODGalerkinReducedProblem
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearReducedProblem, PODGalerkinReducedProblem)
def NonlinearPODGalerkinReducedProblem(ParametrizedReducedDifferentialProblem_DerivedClass):

    @PreserveClassName
    class NonlinearPODGalerkinReducedProblem_Class(ParametrizedReducedDifferentialProblem_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearPODGalerkinReducedProblem_Class
