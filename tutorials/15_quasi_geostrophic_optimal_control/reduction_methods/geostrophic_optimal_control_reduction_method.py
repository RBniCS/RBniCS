# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import LinearReductionMethod


def GeostrophicOptimalControlReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    GeostrophicOptimalControlReductionMethod_Base = LinearReductionMethod(
        DifferentialProblemReductionMethod_DerivedClass)

    class GeostrophicOptimalControlReductionMethod_Class(GeostrophicOptimalControlReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return GeostrophicOptimalControlReductionMethod_Class
