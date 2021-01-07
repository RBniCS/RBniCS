# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic.elliptic_problem import EllipticProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearPODGalerkinReduction
from rbnics.reduction_methods.elliptic.elliptic_reduction_method import EllipticReductionMethod
from rbnics.utils.decorators import ReductionMethodFor

EllipticPODGalerkinReduction_Base = LinearPODGalerkinReduction(
    EllipticReductionMethod(DifferentialProblemReductionMethod))


# Base class containing the interface of a POD-Galerkin ROM
# for elliptic problems
@ReductionMethodFor(EllipticProblem, "PODGalerkin")
class EllipticPODGalerkinReduction(EllipticPODGalerkinReduction_Base):
    pass
