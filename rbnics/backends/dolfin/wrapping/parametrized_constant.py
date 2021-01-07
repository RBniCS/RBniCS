# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

import re
from dolfin import Expression


def ParametrizedConstant(truth_problem, parametrized_constant_code=None, *args, **kwargs):
    from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression
    if "element" not in kwargs:
        element = truth_problem.V.ufl_element()
        while element.num_sub_elements() > 0:
            element = element.sub_elements()[0]
        kwargs["element"] = element
    return ParametrizedExpression(truth_problem, parametrized_constant_code, *args, **kwargs)


def is_parametrized_constant(expr):
    if not isinstance(expr, Expression):
        return False
    else:
        cppcode = expr._cppcode
        return bool(is_parametrized_constant.regex.match(cppcode))


is_parametrized_constant.regex = re.compile("^mu_[0-9]+$")


def parametrized_constant_to_float(expr, point=None):
    if point is None:
        point = expr._mesh.coordinates()[0]
    return float(expr(point))


def expression_float(self):
    if is_parametrized_constant(self):
        return parametrized_constant_to_float(self)
    else:
        return NotImplemented


Expression.__float__ = expression_float
