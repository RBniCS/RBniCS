# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.parabolic.parabolic_coercive_problem import ParabolicCoerciveProblem
from rbnics.problems.parabolic.parabolic_problem import ParabolicProblem
from rbnics.reduction_methods.base import LinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.reduction_methods.parabolic.parabolic_reduction_method import ParabolicReductionMethod

ParabolicPODGalerkinReduction_Base = LinearTimeDependentPODGalerkinReduction(ParabolicReductionMethod(
    EllipticPODGalerkinReduction))


# Base class containing the interface of a POD-Galerkin ROM
# for parabolic problems
@ReductionMethodFor(ParabolicProblem, "PODGalerkin")
@ReductionMethodFor(ParabolicCoerciveProblem, "PODGalerkin")
class ParabolicPODGalerkinReduction(ParabolicPODGalerkinReduction_Base):
    pass
