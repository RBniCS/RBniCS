# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticRBReducedProblem
from rbnics.problems.parabolic.parabolic_reduced_problem import ParabolicReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.parabolic.abstract_parabolic_rb_reduced_problem import AbstractParabolicRBReducedProblem
from rbnics.problems.parabolic.parabolic_problem import ParabolicProblem
from rbnics.reduction_methods.parabolic import ParabolicRBReduction

ParabolicRBReducedProblem_Base = AbstractParabolicRBReducedProblem(ParabolicReducedProblem(EllipticRBReducedProblem))


# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@ReducedProblemFor(ParabolicProblem, ParabolicRBReduction)
class ParabolicRBReducedProblem(ParabolicRBReducedProblem_Base):
    pass
