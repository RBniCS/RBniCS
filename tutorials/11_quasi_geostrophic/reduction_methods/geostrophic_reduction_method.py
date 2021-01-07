# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import LinearReductionMethod


# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
def GeostrophicReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    GeostrophicReductionMethod_Base = LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class GeostrophicReductionMethod_Class(GeostrophicReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return GeostrophicReductionMethod_Class
