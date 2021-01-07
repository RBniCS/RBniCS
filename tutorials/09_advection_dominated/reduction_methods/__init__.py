# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from .online_rectification_decorated_reduction_method import OnlineRectificationDecoratedReductionMethod
from .online_stabilization_decorated_reduction_method import OnlineStabilizationDecoratedReductionMethod
from .online_vanishing_viscosity_decorated_reduction_method import OnlineVanishingViscosityDecoratedReductionMethod

__all__ = [
    "OnlineRectificationDecoratedReductionMethod",
    "OnlineStabilizationDecoratedReductionMethod",
    "OnlineVanishingViscosityDecoratedReductionMethod"
]
