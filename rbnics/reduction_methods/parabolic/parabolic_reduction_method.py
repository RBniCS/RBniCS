# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import TimeDependentReductionMethod


# Base class containing the interface of a projection based ROM
# for parabolic problems.
def ParabolicReductionMethod(EllipticReductionMethod_DerivedClass):

    ParabolicReductionMethod_Base = TimeDependentReductionMethod(EllipticReductionMethod_DerivedClass)

    class ParabolicReductionMethod_Class(ParabolicReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return ParabolicReductionMethod_Class
