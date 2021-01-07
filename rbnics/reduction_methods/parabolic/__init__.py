# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.parabolic.parabolic_pod_galerkin_reduction import ParabolicPODGalerkinReduction
from rbnics.reduction_methods.parabolic.parabolic_rb_reduction import ParabolicRBReduction
from rbnics.reduction_methods.parabolic.parabolic_reduction_method import ParabolicReductionMethod

__all__ = [
    "ParabolicPODGalerkinReduction",
    "ParabolicRBReduction",
    "ParabolicReductionMethod"
]
