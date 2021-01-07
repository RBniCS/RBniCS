# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import PreserveClassName, ReductionMethodDecoratorFor
from problems import OnlineStabilization


@ReductionMethodDecoratorFor(OnlineStabilization)
def OnlineStabilizationDecoratedReductionMethod(EllipticCoerciveReductionMethod_DerivedClass):

    @PreserveClassName
    class OnlineStabilizationDecoratedReductionMethod_Class(EllipticCoerciveReductionMethod_DerivedClass):
        pass

    # return value (a class) for the decorator
    return OnlineStabilizationDecoratedReductionMethod_Class
