# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later


def EllipticCoerciveReducedProblem(EllipticReducedProblem_DerivedClass):

    EllipticCoerciveReducedProblem_Base = EllipticReducedProblem_DerivedClass

    # Base class containing the interface of a projection based ROM
    # for elliptic coercive problems.
    class EllipticCoerciveReducedProblem_Class(EllipticCoerciveReducedProblem_Base):
        pass

    # return value (a class) for the decorator
    return EllipticCoerciveReducedProblem_Class
