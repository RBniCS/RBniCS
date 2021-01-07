# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from rbnics.backends.dolfin.wrapping.compute_theta_for_restriction import compute_theta_for_restriction
from rbnics.utils.decorators import overload


def compute_theta_for_supremizers(compute_theta):
    from rbnics.problems.stokes import StokesProblem
    from rbnics.problems.stokes_optimal_control import StokesOptimalControlProblem

    module = types.ModuleType("compute_theta_for_supremizers",
                              "Storage for implementation of compute_theta_for_supremizers")

    def compute_theta_for_supremizers_impl(self, term):
        return module._compute_theta_for_supremizers_impl(self, term)

    # Stokes problem
    @overload(StokesProblem, str, module=module)
    def _compute_theta_for_supremizers_impl(self_, term):
        return _compute_theta_for_supremizers_impl_stokes_problem(self_, term)

    _compute_theta_for_supremizers_impl_stokes_problem = (
        compute_theta_for_restriction({"bt_restricted": "bt"})(
            compute_theta
        )
    )

    # Stokes optimal control problem
    @overload(StokesOptimalControlProblem, str, module=module)
    def _compute_theta_for_supremizers_impl(self_, term):
        return _compute_theta_for_supremizers_impl_stokes_optimal_control_problem(self_, term)

    _compute_theta_for_supremizers_impl_stokes_optimal_control_problem = (
        compute_theta_for_restriction({"bt*_restricted": "bt*"})(
            compute_theta_for_restriction({"bt_restricted": "bt"})(
                compute_theta
            )
        )
    )

    return compute_theta_for_supremizers_impl
