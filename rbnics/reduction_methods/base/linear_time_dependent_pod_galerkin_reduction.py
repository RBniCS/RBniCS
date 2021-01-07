# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.linear_pod_galerkin_reduction import LinearPODGalerkinReduction
from rbnics.reduction_methods.base.time_dependent_pod_galerkin_reduction import TimeDependentPODGalerkinReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearPODGalerkinReduction, TimeDependentPODGalerkinReduction)
def LinearTimeDependentPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentPODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearTimeDependentPODGalerkinReduction_Class
