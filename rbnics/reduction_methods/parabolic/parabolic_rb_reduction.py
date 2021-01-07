# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.parabolic.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.problems.parabolic.parabolic_problem import ParabolicProblem
from rbnics.reduction_methods.base import LinearTimeDependentRBReduction
from rbnics.reduction_methods.elliptic import EllipticRBReduction
from rbnics.reduction_methods.parabolic.parabolic_reduction_method import ParabolicReductionMethod

ParabolicRBReduction_Base = LinearTimeDependentRBReduction(ParabolicReductionMethod(EllipticRBReduction))


# Base class containing the interface of a RB ROM
# for parabolic problems
@ReductionMethodFor(ParabolicProblem, "ReducedBasis")
@ReductionMethodFor(ParabolicCoerciveProblem, "ReducedBasis")
class ParabolicRBReduction(ParabolicRBReduction_Base):
    pass
