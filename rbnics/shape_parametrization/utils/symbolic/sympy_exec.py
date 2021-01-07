# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import math
from numbers import Number
import sympy


def sympy_exec(string, locals):
    locals = dict(locals)
    locals.update(math_sympy_locals)
    exec(string, {"__builtins__": None}, locals)  # stores the result in an expression named e
    return e  # noqa: F821


math_sympy_locals = dict()
for package in (math, sympy):
    for name, item in package.__dict__.items():
        if callable(item) or isinstance(item, Number):
            math_sympy_locals[name] = item
