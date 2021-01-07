# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.stokes_optimal_control.stokes_optimal_control_reduced_problem import (
    StokesOptimalControlReducedProblem)
from rbnics.utils.decorators import ReducedProblemFor
from rbnics.problems.stokes_optimal_control.stokes_optimal_control_problem import StokesOptimalControlProblem
from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.reduction_methods.stokes_optimal_control import StokesOptimalControlPODGalerkinReduction

StokesOptimalControlPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    StokesOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem))


@ReducedProblemFor(StokesOptimalControlProblem, StokesOptimalControlPODGalerkinReduction)
class StokesOptimalControlPODGalerkinReducedProblem(StokesOptimalControlPODGalerkinReducedProblem_Base):
    pass
