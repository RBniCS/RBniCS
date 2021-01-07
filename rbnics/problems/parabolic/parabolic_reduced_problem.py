# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.parabolic.abstract_parabolic_reduced_problem import AbstractParabolicReducedProblem


# Base class containing the interface of a projection based ROM
# for parabolic problems.
def ParabolicReducedProblem(EllipticReducedProblem_DerivedClass):

    ParabolicReducedProblem_Base = AbstractParabolicReducedProblem(EllipticReducedProblem_DerivedClass)

    class ParabolicReducedProblem_Class(ParabolicReducedProblem_Base):
        pass

    # return value (a class) for the decorator
    return ParabolicReducedProblem_Class
