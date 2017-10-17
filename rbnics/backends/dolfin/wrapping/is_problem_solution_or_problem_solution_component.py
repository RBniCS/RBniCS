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

from ufl.core.multiindex import FixedIndex, Index, MultiIndex
from ufl.indexed import Indexed
from dolfin import Function, split
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import _get_sub_elements__recursive
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map

def is_problem_solution_or_problem_solution_component(node):
    _prepare_solution_split_storage()
    node = _remove_mute_indices(node)
    return node in _solution_split_to_component
    
def _prepare_solution_split_storage():
    for solution in _solution_to_problem_map:
        if solution not in _solution_split_to_component:
            assert solution not in _solution_split_to_solution
            _split_function(solution, _solution_split_to_component, _solution_split_to_solution)
            
def _split_function(solution, solution_split_to_component, solution_split_to_solution):
    solution_split_to_component[solution] = (None, )
    solution_split_to_solution[solution] = solution
    sub_elements = _get_all_sub_elements(solution.function_space())
    for sub_element_index in sub_elements:
        sub_solution = _split_from_tuple(solution, sub_element_index)
        solution_split_to_component[sub_solution] = sub_element_index
        solution_split_to_solution[sub_solution] = solution
                
def _remove_mute_indices(node):
    if isinstance(node, Indexed):
        assert len(node.ufl_operands) == 2
        assert isinstance(node.ufl_operands[0], Function)
        assert isinstance(node.ufl_operands[1], MultiIndex)
        indices = node.ufl_operands[1].indices()
        is_fixed = isinstance(indices[0], FixedIndex)
        assert all([isinstance(index, FixedIndex) == is_fixed for index in indices])
        is_mute = isinstance(indices[0], Index) # mute index for sum
        assert all([isinstance(index, Index) == is_mute for index in indices])
        assert (is_fixed and not is_mute) or (not is_fixed and is_mute)
        if is_fixed:
            return node
        elif is_mute:
            return node.ufl_operands[0]
        else:
            raise TypeError("Invalid index")
    else:
        return node
    
# the difference between this function and the one in function_extend_or_restrict is that the
# _get_sub_elements() in function_extend_or_restrict.py stores only the leaves of the elements tree, while
# _get_all_sub_elements() in this file stores both internal nodes and leaves
def _get_all_sub_elements(V):
    return _get_all_sub_elements__recursive(V, None)
    
def _get_all_sub_elements__recursive(V, index_V):
    sub_elements = dict()
    if V.num_sub_spaces() == 0:
        if index_V is not None:
            sub_elements[tuple(index_V)] = V.ufl_element()
        return sub_elements
    else:
        for i in range(V.num_sub_spaces()):
            index_V_comma_i = list()
            if index_V is not None:
                index_V_comma_i.extend(index_V)
            index_V_comma_i.append(i)
            sub_elements_i = _get_sub_elements__recursive(V.sub(i), index_V_comma_i)
            sub_elements.update(sub_elements_i)
            sub_elements[tuple(index_V_comma_i)] = V.ufl_element()
        return sub_elements
    
def _split_from_tuple(input_, index_as_tuple):
    assert isinstance(index_as_tuple, tuple)
    assert len(index_as_tuple) > 0
    if len(index_as_tuple) == 1 and index_as_tuple[0] is None:
        return input_
    else:
        for i in index_as_tuple:
            input_ = split(input_)[i]
        return input_
    
_solution_split_to_component = dict()
_solution_split_to_solution = dict()
