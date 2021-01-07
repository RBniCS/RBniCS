# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from rbnics.backends.dolfin.wrapping.compute_theta_for_derivative import compute_theta_for_derivative
from rbnics.utils.decorators import overload


def compute_theta_for_derivatives(compute_theta):
    from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
    from rbnics.problems.navier_stokes import NavierStokesProblem

    module = types.ModuleType("compute_theta_for_derivatives",
                              "Storage for implementation of compute_theta_for_derivatives")

    def compute_theta_for_derivatives_impl(self, term):
        return module._compute_theta_for_derivatives_impl(self, term)

    # Nonlinear elliptic problem
    @overload(NonlinearEllipticProblem, str, module=module)
    def _compute_theta_for_derivatives_impl(self_, term):
        return _compute_theta_for_derivatives_impl_nonlinear_elliptic_problem(self_, term)

    _compute_theta_for_derivatives_impl_nonlinear_elliptic_problem = (
        compute_theta_for_derivative({"dc": "c"})(
            compute_theta
        )
    )

    # Navier Stokes problem
    @overload(NavierStokesProblem, str, module=module)
    def _compute_theta_for_derivatives_impl(self_, term):
        return _compute_theta_for_derivatives_impl_navier_stokes_problem(self_, term)

    _compute_theta_for_derivatives_impl_navier_stokes_problem = (
        compute_theta_for_derivative({"dc": "c"})(
            compute_theta
        )
    )

    return compute_theta_for_derivatives_impl
