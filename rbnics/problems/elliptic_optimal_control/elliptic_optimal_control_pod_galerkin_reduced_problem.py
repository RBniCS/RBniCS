# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_reduced_problem import (
    EllipticOptimalControlReducedProblem)
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.elliptic_optimal_control.elliptic_optimal_control_problem import EllipticOptimalControlProblem
from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.elliptic_optimal_control import EllipticOptimalControlPODGalerkinReduction

EllipticOptimalControlPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    EllipticOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem))


@ReducedProblemFor(EllipticOptimalControlProblem, EllipticOptimalControlPODGalerkinReduction)
class EllipticOptimalControlPODGalerkinReducedProblem(EllipticOptimalControlPODGalerkinReducedProblem_Base):
    pass
