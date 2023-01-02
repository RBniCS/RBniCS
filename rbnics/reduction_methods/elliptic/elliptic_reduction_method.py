# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import LinearReductionMethod


# Base class containing the interface of a projection based ROM
# for elliptic problems.
def EllipticReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    EllipticReductionMethod_Base = LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class EllipticReductionMethod_Class(EllipticReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return EllipticReductionMethod_Class
