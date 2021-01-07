# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_pod_galerkin_reduction import (
    EllipticOptimalControlPODGalerkinReduction)
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_rb_reduction import (
    EllipticOptimalControlRBReduction)
from rbnics.reduction_methods.elliptic_optimal_control.elliptic_optimal_control_reduction_method import (
    EllipticOptimalControlReductionMethod)

__all__ = [
    "EllipticOptimalControlPODGalerkinReduction",
    "EllipticOptimalControlRBReduction",
    "EllipticOptimalControlReductionMethod"
]
