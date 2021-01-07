# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.core.operator import Operator
from dolfin import assign, Function, LagrangeInterpolator, project
from dolfin.function.expression import BaseExpression


def evaluate_expression(expression, function, replaced_expression=None):
    if replaced_expression is None:
        replaced_expression = expression
    assert isinstance(expression, (BaseExpression, Function, Operator))
    if isinstance(expression, BaseExpression):
        LagrangeInterpolator.interpolate(function, replaced_expression)
    elif isinstance(expression, Function):
        assign(function, replaced_expression)
    elif isinstance(expression, Operator):
        project(replaced_expression, function.function_space(), function=function)
    else:
        raise ValueError("Invalid expression")
