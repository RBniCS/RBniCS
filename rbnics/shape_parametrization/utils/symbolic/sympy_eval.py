# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import math
from numbers import Number


def sympy_eval(string, locals):
    locals = dict(locals)
    locals.update(math_locals)
    return eval(string, {"__builtins__": None}, locals)


math_locals = dict()
for package in (math, ):
    for name, item in package.__dict__.items():
        if callable(item) or isinstance(item, Number):
            math_locals[name] = item
