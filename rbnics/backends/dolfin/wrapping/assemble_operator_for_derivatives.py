# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from rbnics.backends.dolfin.wrapping.assemble_operator_for_derivative import assemble_operator_for_derivative
from rbnics.utils.decorators import overload


def assemble_operator_for_derivatives(assemble_operator):
    from rbnics.problems.nonlinear_elliptic import NonlinearEllipticProblem
    from rbnics.problems.navier_stokes import NavierStokesProblem

    module = types.ModuleType("assemble_operator_for_derivatives",
                              "Storage for implementation of assemble_operator_for_derivatives")

    def assemble_operator_for_derivatives_impl(self, term):
        return module._assemble_operator_for_derivatives_impl(self, term)

    # Nonlinear elliptic problem
    @overload(NonlinearEllipticProblem, str, module=module)
    def _assemble_operator_for_derivatives_impl(self_, term):
        return _assemble_operator_for_derivatives_impl_nonlinear_elliptic_problem(self_, term)

    _assemble_operator_for_derivatives_impl_nonlinear_elliptic_problem = (
        assemble_operator_for_derivative({"dc": "c"})(
            assemble_operator
        )
    )

    # Navier Stokes problem
    @overload(NavierStokesProblem, str, module=module)
    def _assemble_operator_for_derivatives_impl(self_, term):
        return _assemble_operator_for_derivatives_impl_navier_stokes_problem(self_, term)

    _assemble_operator_for_derivatives_impl_navier_stokes_problem = (
        assemble_operator_for_derivative({"dc": "c"})(
            assemble_operator
        )
    )

    return assemble_operator_for_derivatives_impl
