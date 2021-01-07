# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.shape_parametrization.utils.symbolic.strings_to_number_of_parameters import strings_to_number_of_parameters


def strings_to_sympy_symbolic_parameters(strings, SymbolGenerator):
    P = strings_to_number_of_parameters(strings)
    if P > 0:
        mu = SymbolGenerator("mu", P, 1)
    else:
        mu = None
    return mu
