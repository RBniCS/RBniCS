# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.nonlinear_parabolic.nonlinear_parabolic_pod_galerkin_reduction import (
    NonlinearParabolicPODGalerkinReduction)
# from rbnics.reduction_methods.nonlinear_parabolic.nonlinear_parabolic_rb_reduction import (
#    NonlinearParabolicRBReduction)
from rbnics.reduction_methods.nonlinear_parabolic.nonlinear_parabolic_reduction_method import (
    NonlinearParabolicReductionMethod)

__all__ = [
    "NonlinearParabolicPODGalerkinReduction",
    # "NonlinearParabolicRBReduction",
    "NonlinearParabolicReductionMethod"
]
