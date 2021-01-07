# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl import Argument
from ufl.constantvalue import ConstantValue
from ufl.core.operator import Operator
from ufl.core.multiindex import IndexBase, MultiIndex
from ufl.geometry import GeometricQuantity
from ufl.indexed import Indexed
from ufl.tensors import ListTensor
from dolfin import Constant, Function, MixedElement, split, TensorElement
from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import _get_sub_elements__recursive
from rbnics.utils.cache import Cache
from rbnics.utils.decorators import overload
from rbnics.utils.decorators.store_map_from_solution_to_problem import _solution_to_problem_map


def is_problem_solution(node):
    _prepare_solution_split_storage()
    node = _remove_all_indices(node)
    return node in _solution_split_to_component


def _prepare_solution_split_storage():
    for solution in _solution_to_problem_map:
        if solution not in _solution_split_to_component:
            assert solution not in _solution_split_to_solution
            _split_function(solution, _solution_split_to_component, _solution_split_to_solution)


def _split_function(solution, solution_split_to_component, solution_split_to_solution):
    solution_split_to_component[solution] = (None, )
    solution_split_to_solution[solution] = solution
    element = solution.ufl_element()
    if (
        isinstance(element, MixedElement)
        and not isinstance(element, TensorElement)  # split() does not work with TensorElement
    ):
        sub_elements = _get_all_sub_elements(solution.function_space())
        for sub_element_index in sub_elements:
            sub_solution = _split_from_tuple(solution, sub_element_index)
            solution_split_to_component[sub_solution] = sub_element_index
            solution_split_to_solution[sub_solution] = solution


@overload
def _remove_all_indices(node: (Argument, BaseExpression, Constant, ConstantValue, Function, GeometricQuantity,
                               IndexBase, MultiIndex, Operator)):
    return node


@overload
def _remove_all_indices(node: Indexed):
    assert len(node.ufl_operands) == 2
    assert isinstance(node.ufl_operands[1], MultiIndex)
    return _remove_all_indices(node.ufl_operands[0])


@overload
def _remove_all_indices(node: ListTensor):
    output = {_remove_all_indices(operand) for operand in node.ufl_operands}
    assert len(output) == 1
    return output.pop()


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


_solution_split_to_component = Cache()
_solution_split_to_solution = Cache()
