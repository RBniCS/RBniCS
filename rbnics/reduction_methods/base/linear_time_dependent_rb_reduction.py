# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.linear_rb_reduction import LinearRBReduction
from rbnics.reduction_methods.base.time_dependent_rb_reduction import TimeDependentRBReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearRBReduction, TimeDependentRBReduction)
def LinearTimeDependentRBReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class LinearTimeDependentRBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearTimeDependentRBReduction_Class
