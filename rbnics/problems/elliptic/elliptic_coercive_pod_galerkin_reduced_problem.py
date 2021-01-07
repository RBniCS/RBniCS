# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic.elliptic_coercive_problem import EllipticCoerciveProblem
from rbnics.problems.elliptic.elliptic_coercive_reduced_problem import EllipticCoerciveReducedProblem
from rbnics.problems.elliptic.elliptic_pod_galerkin_reduced_problem import EllipticPODGalerkinReducedProblem
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticCoercivePODGalerkinReducedProblem_Base = EllipticCoerciveReducedProblem(EllipticPODGalerkinReducedProblem)


# Base class containing the interface of a projection based ROM
# for elliptic coercive problems.
@ReducedProblemFor(EllipticCoerciveProblem, EllipticPODGalerkinReduction)
class EllipticCoercivePODGalerkinReducedProblem(EllipticCoercivePODGalerkinReducedProblem_Base):
    pass
