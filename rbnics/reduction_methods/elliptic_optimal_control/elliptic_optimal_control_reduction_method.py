# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.base import LinearReductionMethod


# Base class containing the interface of a projection based ROM
# for saddle point problems.
def EllipticOptimalControlReductionMethod(DifferentialProblemReductionMethod_DerivedClass):

    EllipticOptimalControlReductionMethod_Base = LinearReductionMethod(DifferentialProblemReductionMethod_DerivedClass)

    class EllipticOptimalControlReductionMethod_Class(EllipticOptimalControlReductionMethod_Base):
        pass

    # return value (a class) for the decorator
    return EllipticOptimalControlReductionMethod_Class
