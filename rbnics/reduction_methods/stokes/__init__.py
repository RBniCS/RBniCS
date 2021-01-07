# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.stokes.stokes_pod_galerkin_reduction import StokesPODGalerkinReduction
from rbnics.reduction_methods.stokes.stokes_rb_reduction import StokesRBReduction
from rbnics.reduction_methods.stokes.stokes_reduction_method import StokesReductionMethod

__all__ = [
    "StokesPODGalerkinReduction",
    "StokesRBReduction",
    "StokesReductionMethod"
]
