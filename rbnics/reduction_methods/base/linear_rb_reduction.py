# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base.linear_reduction_method import LinearReductionMethod
from rbnics.reduction_methods.base.rb_reduction import RBReduction
from rbnics.utils.decorators import PreserveClassName, RequiredBaseDecorators


@RequiredBaseDecorators(LinearReductionMethod, RBReduction)
def LinearRBReduction(DifferentialProblemReductionMethod_DerivedClass):

    @PreserveClassName
    class LinearRBReduction_Class(DifferentialProblemReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return LinearRBReduction_Class
