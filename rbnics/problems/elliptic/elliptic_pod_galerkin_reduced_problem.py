# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.problems.elliptic.elliptic_problem import EllipticProblem
from rbnics.problems.elliptic.elliptic_reduced_problem import EllipticReducedProblem
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    EllipticReducedProblem(ParametrizedReducedDifferentialProblem))


# Base class containing the interface of a projection based ROM
# for elliptic problems.
@ReducedProblemFor(EllipticProblem, EllipticPODGalerkinReduction)
class EllipticPODGalerkinReducedProblem(EllipticPODGalerkinReducedProblem_Base):
    pass
