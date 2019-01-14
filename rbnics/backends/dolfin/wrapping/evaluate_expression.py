# Copyright (C) 2015-2019 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

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
