# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.nonlinear_pod_galerkin_reduction import NonlinearPODGalerkinReduction
from rbnics.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearPODGalerkinReduction, TimeDependentPODGalerkinReduction)
def NonlinearTimeDependentPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class NonlinearTimeDependentPODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearTimeDependentPODGalerkinReduction_Class
