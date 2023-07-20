# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
try:
    from ufl_legacy import Form
except ImportError:
    from ufl import Form
from dolfin import Constant


original__mul__ = Form.__mul__


def custom__mul__(self, other):
    if isinstance(other, Number):
        return self.__rmul__(Constant(other))
    else:
        return original__mul__(self, other)


setattr(Form, "__mul__", custom__mul__)


original__rmul__ = Form.__rmul__


def custom__rmul__(self, other):
    if isinstance(other, Number):
        return original__rmul__(self, Constant(other))
    else:
        return original__rmul__(self, other)


setattr(Form, "__rmul__", custom__rmul__)


def custom__truediv__(self, other):
    if isinstance(other, Number):
        return self.__rmul__(Constant(1. / other))
    else:
        return NotImplemented


setattr(Form, "__truediv__", custom__truediv__)
