# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticPODGalerkinReducedProblem
from rbnics.problems.parabolic.parabolic_reduced_problem import ParabolicReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.parabolic.abstract_parabolic_pod_galerkin_reduced_problem import (
    AbstractParabolicPODGalerkinReducedProblem)
from rbnics.problems.parabolic.parabolic_problem import ParabolicProblem
from rbnics.reduction_methods.parabolic import ParabolicPODGalerkinReduction

ParabolicPODGalerkinReducedProblem_Base = AbstractParabolicPODGalerkinReducedProblem(
    ParabolicReducedProblem(EllipticPODGalerkinReducedProblem))


# Base class containing the interface of a projection based ROM
# for parabolic problems.
@ReducedProblemFor(ParabolicProblem, ParabolicPODGalerkinReduction)
class ParabolicPODGalerkinReducedProblem(ParabolicPODGalerkinReducedProblem_Base):
    pass
