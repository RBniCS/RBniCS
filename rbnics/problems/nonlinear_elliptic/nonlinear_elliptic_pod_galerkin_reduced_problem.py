# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic import EllipticPODGalerkinReducedProblem
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_reduced_problem import NonlinearEllipticReducedProblem
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.base import NonlinearPODGalerkinReducedProblem
from rbnics.problems.nonlinear_elliptic.nonlinear_elliptic_problem import NonlinearEllipticProblem
from rbnics.reduction_methods.nonlinear_elliptic import NonlinearEllipticPODGalerkinReduction

NonlinearEllipticPODGalerkinReducedProblem_Base = NonlinearPODGalerkinReducedProblem(
    NonlinearEllipticReducedProblem(EllipticPODGalerkinReducedProblem))


@ReducedProblemFor(NonlinearEllipticProblem, NonlinearEllipticPODGalerkinReduction)
class NonlinearEllipticPODGalerkinReducedProblem(NonlinearEllipticPODGalerkinReducedProblem_Base):
    pass
