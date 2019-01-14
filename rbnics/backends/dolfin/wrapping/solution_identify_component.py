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
from ufl.core.multiindex import FixedIndex, Index as MuteIndex, IndexBase, MultiIndex
from ufl.geometry import GeometricQuantity
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Constant, Function
from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.is_problem_solution import _solution_split_to_component, _solution_split_to_solution
from rbnics.utils.decorators import overload

def solution_identify_component(node):
    node = _remove_mute_indices(node)
    return _solution_identify_component(node)
    
def _solution_identify_component(node):
    assert node in _solution_split_to_component
    assert node in _solution_split_to_solution
    return (node, _solution_split_to_component[node], _solution_split_to_solution[node])
    
@overload
def _remove_mute_indices(node: (Argument, BaseExpression, Constant, ConstantValue, Function, GeometricQuantity, IndexBase, MultiIndex, Operator)):
    return node
    
@overload
def _remove_mute_indices(node: Indexed):
    assert len(node.ufl_operands) == 2
    assert isinstance(node.ufl_operands[1], MultiIndex)
    indices = node.ufl_operands[1].indices()
    is_fixed = isinstance(indices[0], FixedIndex)
    assert all([isinstance(index, FixedIndex) == is_fixed for index in indices])
    is_mute = isinstance(indices[0], MuteIndex) # mute index for sum
    assert all([isinstance(index, MuteIndex) == is_mute for index in indices])
    assert (is_fixed and not is_mute) or (not is_fixed and is_mute)
    if is_fixed:
        return node
    elif is_mute:
        return _remove_mute_indices(node.ufl_operands[0])
    else:
        raise TypeError("Invalid index")
        
@overload
def _remove_mute_indices(node: ListTensor):
    return node._ufl_expr_reconstruct_(*[_remove_mute_indices(operand) for operand in node.ufl_operands])
