# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .geostrophic_reduction_method import GeostrophicReductionMethod
from .geostrophic_pod_galerkin_reduction import GeostrophicPODGalerkinReduction

__all__ = [
    "GeostrophicReductionMethod",
    "GeostrophicPODGalerkinReduction"
]
