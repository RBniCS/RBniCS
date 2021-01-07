# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from rbnics.utils.decorators import overload


def compute_theta_for_stability_factor(compute_theta):
    from rbnics.problems.elliptic import EllipticCoerciveProblem

    module = types.ModuleType("compute_theta_for_stability_factor",
                              "Storage for implementation of compute_theta_for_stability_factor")

    def compute_theta_for_stability_factor_impl(self, term):
        return module._compute_theta_for_stability_factor_impl(self, term)

    # Elliptic coercive problem
    @overload(EllipticCoerciveProblem, str, module=module)
    def _compute_theta_for_stability_factor_impl(self_, term):
        if term == "stability_factor_left_hand_matrix":
            return tuple(0.5 * t for t in compute_theta(self_, "a"))
        else:
            return compute_theta(self_, term)

    return compute_theta_for_stability_factor_impl
