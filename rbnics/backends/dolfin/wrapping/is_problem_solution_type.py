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

from ufl import Argument
from ufl.constantvalue import ConstantValue
from ufl.core.operator import Operator
from ufl.core.multiindex import IndexBase, MultiIndex
from ufl.geometry import GeometricQuantity
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Constant, Function
from dolfin.function.expression import BaseExpression
from rbnics.utils.decorators import overload

@overload
def is_problem_solution_type(node: (Argument, BaseExpression, Constant, ConstantValue, GeometricQuantity, IndexBase, MultiIndex, Operator)):
    return False
    
@overload
def is_problem_solution_type(node: Function):
    return True
    
@overload
def is_problem_solution_type(node: Indexed):
    assert len(node.ufl_operands) == 2
    assert isinstance(node.ufl_operands[1], MultiIndex)
    return is_problem_solution_type(node.ufl_operands[0])
    
@overload
def is_problem_solution_type(node: ListTensor):
    result = [is_problem_solution_type(component) for component in node.ufl_operands]
    assert all([result_i == result[0] for result_i in result[1:]])
    return result[0]
