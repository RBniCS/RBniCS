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
    return expr(point)
    
def expression_float(self):
    if is_parametrized_constant(self):
        return parametrized_constant_to_float(self)
    else:
        return NotImplemented
Expression.__float__ = expression_float
