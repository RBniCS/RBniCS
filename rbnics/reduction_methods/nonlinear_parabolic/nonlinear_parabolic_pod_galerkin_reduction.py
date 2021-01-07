# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.utils.decorators import ReductionMethodFor
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_problem import NonlinearParabolicProblem
from rbnics.reduction_methods.base import NonlinearTimeDependentPODGalerkinReduction
from rbnics.reduction_methods.nonlinear_elliptic import NonlinearEllipticPODGalerkinReduction
from rbnics.reduction_methods.nonlinear_parabolic.nonlinear_parabolic_reduction_method import (
    NonlinearParabolicReductionMethod)

NonlinearParabolicPODGalerkinReduction_Base = NonlinearTimeDependentPODGalerkinReduction(
    NonlinearParabolicReductionMethod(NonlinearEllipticPODGalerkinReduction))


@ReductionMethodFor(NonlinearParabolicProblem, "PODGalerkin")
class NonlinearParabolicPODGalerkinReduction(NonlinearParabolicPODGalerkinReduction_Base):
    pass
