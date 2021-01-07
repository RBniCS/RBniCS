# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import types
from dolfin import FunctionSpace
from rbnics.utils.decorators import overload


def generate_function_space_for_stability_factor(__init__):
    from rbnics.problems.elliptic import EllipticCoerciveProblem
    from rbnics.problems.stokes import StokesProblem

    module = types.ModuleType("generate_function_space_for_stability_factor",
                              "Storage for implementation of generate_function_space_for_stability_factor")

    def generate_function_space_for_stability_factor_impl(self, V, **kwargs):
        __init__(self, V, **kwargs)
        module._generate_function_space_for_stability_factor_impl(self, V)

    # Elliptic coercive or Stokes problems
    @overload((EllipticCoerciveProblem, StokesProblem), FunctionSpace, module=module)
    def _generate_function_space_for_stability_factor_impl(self_, V):
        self_.stability_factor_V = V

    return generate_function_space_for_stability_factor_impl
