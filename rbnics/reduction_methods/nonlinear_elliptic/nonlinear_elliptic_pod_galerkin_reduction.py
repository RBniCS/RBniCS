# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.reduction_methods.base import NonlinearPODGalerkinReduction
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.reduction_methods.nonlinear_elliptic.nonlinear_elliptic_reduction_method import (
    NonlinearEllipticReductionMethod)

NonlinearEllipticPODGalerkinReduction_Base = NonlinearPODGalerkinReduction(
    NonlinearEllipticReductionMethod(EllipticPODGalerkinReduction))


@ReductionMethodFor(NonlinearEllipticProblem, "PODGalerkin")
class NonlinearEllipticPODGalerkinReduction(NonlinearEllipticPODGalerkinReduction_Base):
    pass
