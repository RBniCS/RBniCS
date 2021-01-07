# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.reduction_methods.nonlinear_elliptic.nonlinear_elliptic_pod_galerkin_reduction import (
    NonlinearEllipticPODGalerkinReduction)
# from rbnics.reduction_methods.nonlinear_elliptic.nonlinear_elliptic_rb_reduction import NonlinearEllipticRBReduction
from rbnics.reduction_methods.nonlinear_elliptic.nonlinear_elliptic_reduction_method import (
    NonlinearEllipticReductionMethod)

__all__ = [
    "NonlinearEllipticPODGalerkinReduction",
    # "NonlinearEllipticRBReduction",
    "NonlinearEllipticReductionMethod"
]
