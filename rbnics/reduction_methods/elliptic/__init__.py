# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.elliptic.elliptic_pod_galerkin_reduction import EllipticPODGalerkinReduction
from rbnics.reduction_methods.elliptic.elliptic_rb_reduction import EllipticRBReduction
from rbnics.reduction_methods.elliptic.elliptic_reduction_method import EllipticReductionMethod

__all__ = [
    "EllipticPODGalerkinReduction",
    "EllipticRBReduction",
    "EllipticReductionMethod"
]
