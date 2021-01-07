# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.navier_stokes.navier_stokes_pod_galerkin_reduction import NavierStokesPODGalerkinReduction
# from rbnics.reduction_methods.navier_stokes.navier_stokes_rb_reduction import NavierStokesRBReduction
from rbnics.reduction_methods.navier_stokes.navier_stokes_reduction_method import NavierStokesReductionMethod

__all__ = [
    "NavierStokesPODGalerkinReduction",
    # "NavierStokesRBReduction",
    "NavierStokesReductionMethod"
]
