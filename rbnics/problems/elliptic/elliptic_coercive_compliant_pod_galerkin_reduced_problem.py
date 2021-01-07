# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic.elliptic_coercive_compliant_problem import EllipticCoerciveCompliantProblem
from rbnics.problems.elliptic.elliptic_coercive_compliant_reduced_problem import EllipticCoerciveCompliantReducedProblem
from rbnics.problems.elliptic.elliptic_coercive_pod_galerkin_reduced_problem import (
    EllipticCoercivePODGalerkinReducedProblem)
from rbnics.reduction_methods.elliptic import EllipticPODGalerkinReduction
from rbnics.utils.decorators import ReducedProblemFor

EllipticCoerciveCompliantPODGalerkinReducedProblem_Base = EllipticCoerciveCompliantReducedProblem(
    EllipticCoercivePODGalerkinReducedProblem)


# Base class containing the interface of a projection based ROM
# for elliptic coercive compliant problems.
@ReducedProblemFor(EllipticCoerciveCompliantProblem, EllipticPODGalerkinReduction)
class EllipticCoerciveCompliantPODGalerkinReducedProblem(EllipticCoerciveCompliantPODGalerkinReducedProblem_Base):
    pass
