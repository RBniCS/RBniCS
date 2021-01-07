# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import Function


def get_function_space(function):
    assert isinstance(function, Function)
    return function.function_space()
