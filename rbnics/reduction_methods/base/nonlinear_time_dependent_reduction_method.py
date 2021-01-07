# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.nonlinear_reduction_method import NonlinearReductionMethod
from rbnics.reduction_methods.base.time_dependent_reduction_method import TimeDependentReductionMethod
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearReductionMethod, TimeDependentReductionMethod)
def NonlinearTimeDependentReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class NonlinearTimeDependentReductionMethod_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearTimeDependentReductionMethod_Class
