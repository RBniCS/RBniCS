# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.nonlinear_rb_reduction import NonlinearRBReduction
from rbnics.reduction_methods.base.time_dependent_rb_reduction import TimeDependentRBReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(NonlinearRBReduction, TimeDependentRBReduction)
def NonlinearTimeDependentRBReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class NonlinearTimeDependentRBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return NonlinearTimeDependentRBReduction_Class
