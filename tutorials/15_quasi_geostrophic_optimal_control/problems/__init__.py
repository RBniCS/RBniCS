# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .geostrophic_optimal_control_problem import GeostrophicOptimalControlProblem
from .geostrophic_optimal_control_reduced_problem import GeostrophicOptimalControlReducedProblem
from .geostrophic_optimal_control_pod_galerkin_reduced_problem import GeostrophicOptimalControlPODGalerkinReducedProblem

__all__ = [
    "GeostrophicOptimalControlProblem",
    "GeostrophicOptimalControlReducedProblem",
    "GeostrophicOptimalControlPODGalerkinReducedProblem"
]
