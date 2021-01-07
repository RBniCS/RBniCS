# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_pod_galerkin_reduction import (
    StokesOptimalControlPODGalerkinReduction)
# from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_rb_reduction import (
#    StokesOptimalControlRBReduction)
from rbnics.reduction_methods.stokes_optimal_control.stokes_optimal_control_reduction_method import (
    StokesOptimalControlReductionMethod)

__all__ = [
    "StokesOptimalControlPODGalerkinReduction",
    # "StokesOptimalControlRBReduction",
    "StokesOptimalControlReductionMethod"
]
