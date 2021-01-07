# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.nonlinear_elliptic import NonlinearEllipticPODGalerkinReducedProblem
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_reduced_problem import NonlinearParabolicReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearTimeDependentPODGalerkinReducedProblem
from rbnics.problems.nonlinear_parabolic.nonlinear_parabolic_problem import NonlinearParabolicProblem
from rbnics.reduction_methods.nonlinear_parabolic import NonlinearParabolicPODGalerkinReduction

NonlinearParabolicPODGalerkinReducedProblem_Base = NonlinearTimeDependentPODGalerkinReducedProblem(
    NonlinearParabolicReducedProblem(NonlinearEllipticPODGalerkinReducedProblem))


@ReducedProblemFor(NonlinearParabolicProblem, NonlinearParabolicPODGalerkinReduction)
class NonlinearParabolicPODGalerkinReducedProblem(NonlinearParabolicPODGalerkinReducedProblem_Base):
    pass
