# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticCoerciveRBReducedProblem
from rbnics.problems.parabolic.parabolic_coercive_reduced_problem import ParabolicCoerciveReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.parabolic.abstract_parabolic_rb_reduced_problem import AbstractParabolicRBReducedProblem
from rbnics.problems.parabolic.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.reduction_methods.parabolic import ParabolicRBReduction

ParabolicCoerciveRBReducedProblem_Base = AbstractParabolicRBReducedProblem(
    ParabolicCoerciveReducedProblem(EllipticCoerciveRBReducedProblem))


# Base class containing the interface of a projection based ROM
# for parabolic coercive problems.
@ReducedProblemFor(ParabolicCoerciveProblem, ParabolicRBReduction)
class ParabolicCoerciveRBReducedProblem(ParabolicCoerciveRBReducedProblem_Base):
    pass
