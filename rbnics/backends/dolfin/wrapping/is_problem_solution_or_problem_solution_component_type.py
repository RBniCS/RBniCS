# Copyright (C) 2015-2017 by the RBniCS authors
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
from dolfin import Constant, Expression, Function
from rbnics.utils.decorators import overload

@overload
def is_problem_solution_or_problem_solution_component_type(node: (Argument, Constant, ConstantValue, Expression, GeometricQuantity, IndexBase, MultiIndex, Operator)):
    return False
    
@overload
def is_problem_solution_or_problem_solution_component_type(node: Function):
    return True
    
@overload
def is_problem_solution_or_problem_solution_component_type(node: Indexed):
    assert len(node.ufl_operands) == 2
    assert isinstance(node.ufl_operands[0], (Argument, Constant, Expression, Function, Operator))
    assert isinstance(node.ufl_operands[1], MultiIndex)
    return is_problem_solution_or_problem_solution_component_type(node.ufl_operands[0])
    
@overload
def is_problem_solution_or_problem_solution_component_type(node: ListTensor):
    assert all(isinstance(component, Indexed) for component in node.ufl_operands)
    assert all(len(component.ufl_operands) == 2 for component in node.ufl_operands)
    assert all(isinstance(component.ufl_operands[0], (Argument, Function)) for component in node.ufl_operands)
    assert all(isinstance(component.ufl_operands[1], MultiIndex) for component in node.ufl_operands)
    assert all(component.ufl_operands[0] == node.ufl_operands[-1].ufl_operands[0] for component in node.ufl_operands)
    return is_problem_solution_or_problem_solution_component_type(node.ufl_operands[-1].ufl_operands[0])
