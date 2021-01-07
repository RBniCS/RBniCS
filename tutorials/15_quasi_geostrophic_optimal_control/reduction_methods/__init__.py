# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .geostrophic_optimal_control_pod_galerkin_reduction import GeostrophicOptimalControlPODGalerkinReduction
from .geostrophic_optimal_control_reduction_method import GeostrophicOptimalControlReductionMethod

__all__ = [
    "GeostrophicOptimalControlPODGalerkinReduction",
    "GeostrophicOptimalControlReductionMethod"
]
