# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.problems.base import LinearPODGalerkinReducedProblem, ParametrizedReducedDifferentialProblem
from rbnics.utils.decorators import ReducedProblemFor
from reduction_methods import GeostrophicOptimalControlPODGalerkinReduction
from .geostrophic_optimal_control_reduced_problem import GeostrophicOptimalControlReducedProblem
from .geostrophic_optimal_control_problem import GeostrophicOptimalControlProblem

GeostrophicOptimalControlPODGalerkinReducedProblem_Base = LinearPODGalerkinReducedProblem(
    GeostrophicOptimalControlReducedProblem(ParametrizedReducedDifferentialProblem))


@ReducedProblemFor(GeostrophicOptimalControlProblem, GeostrophicOptimalControlPODGalerkinReduction)
class GeostrophicOptimalControlPODGalerkinReducedProblem(GeostrophicOptimalControlPODGalerkinReducedProblem_Base):
    pass
