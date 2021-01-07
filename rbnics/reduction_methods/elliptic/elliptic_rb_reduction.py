# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic.elliptic_problem import EllipticProblem
from rbnics.reduction_methods.base import DifferentialProblemReductionMethod, LinearRBReduction
from rbnics.reduction_methods.elliptic.elliptic_reduction_method import EllipticReductionMethod
from rbnics.utils.decorators import ReductionMethodFor

EllipticRBReduction_Base = LinearRBReduction(EllipticReductionMethod(DifferentialProblemReductionMethod))


# Base class containing the interface of the RB method
# for elliptic problems
@ReductionMethodFor(EllipticProblem, "ReducedBasis")
class EllipticRBReduction(EllipticRBReduction_Base):
    pass
