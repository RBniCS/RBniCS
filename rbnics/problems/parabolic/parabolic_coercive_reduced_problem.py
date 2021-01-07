# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.parabolic.abstract_parabolic_reduced_problem import AbstractParabolicReducedProblem


# Base class containing the interface of a projection based ROM
# for parabolic coercive problems.
def ParabolicCoerciveReducedProblem(EllipticCoerciveReducedProblem_DerivedClass):

    ParabolicCoerciveReducedProblem_Base = AbstractParabolicReducedProblem(EllipticCoerciveReducedProblem_DerivedClass)

    class ParabolicCoerciveReducedProblem_Class(ParabolicCoerciveReducedProblem_Base):
        pass

    # return value (a class) for the decorator
    return ParabolicCoerciveReducedProblem_Class
