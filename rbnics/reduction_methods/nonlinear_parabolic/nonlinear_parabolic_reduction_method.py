# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import TimeDependentReductionMethod


def NonlinearParabolicReductionMethod(NonlinearEllipticReductionMethod_DerivedClass):

    NonlinearParabolicReductionMethod_Base = TimeDependentReductionMethod(NonlinearEllipticReductionMethod_DerivedClass)

    class NonlinearParabolicReductionMethod_Class(NonlinearParabolicReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return NonlinearParabolicReductionMethod_Class
