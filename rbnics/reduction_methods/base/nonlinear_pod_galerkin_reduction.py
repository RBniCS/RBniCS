# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.nonlinear_reduction_method import NonlinearReductionMethod
from rbnics.reduction_methods.base.pod_galerkin_reduction import PODGalerkinReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearReductionMethod, PODGalerkinReduction)
def NonlinearPODGalerkinReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class NonlinearPODGalerkinReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearPODGalerkinReduction_Class
