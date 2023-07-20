# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

try:
    from ufl_legacy import Argument
    from ufl_legacy.constantvalue import ConstantValue
    from ufl_legacy.core.operator import Operator
    from ufl_legacy.core.multiindex import FixedIndex, Index as MuteIndex, IndexBase, MultiIndex
    from ufl_legacy.geometry import GeometricQuantity
    from ufl_legacy.indexed import Indexed
    from ufl_legacy.tensors import ListTensor
except ImportError:
    from ufl import Argument
    from ufl.constantvalue import ConstantValue
    from ufl.core.operator import Operator
    from ufl.core.multiindex import FixedIndex, Index as MuteIndex, IndexBase, MultiIndex
    from ufl.geometry import GeometricQuantity
    from ufl.indexed import Indexed
    from ufl.tensors import ListTensor
from dolfin import Constant, Function
from dolfin.function.expression import BaseExpression
from rbnics.backends.dolfin.wrapping.is_problem_solution import (
    _solution_split_to_component, _solution_split_to_solution)
from rbnics.utils.decorators import overload


def solution_identify_component(node):
    node = _remove_mute_indices(node)
    return _solution_identify_component(node)


def _solution_identify_component(node):
    assert node in _solution_split_to_component
    assert node in _solution_split_to_solution
    return (node, _solution_split_to_component[node], _solution_split_to_solution[node])


@overload
def _remove_mute_indices(node: (Argument, BaseExpression, Constant, ConstantValue, Function, GeometricQuantity,
                                IndexBase, MultiIndex, Operator)):
    return node


@overload
def _remove_mute_indices(node: Indexed):
    assert len(node.ufl_operands) == 2
    assert isinstance(node.ufl_operands[1], MultiIndex)
    indices = node.ufl_operands[1].indices()
    is_fixed = isinstance(indices[0], FixedIndex)
    assert all([isinstance(index, FixedIndex) == is_fixed for index in indices])
    is_mute = isinstance(indices[0], MuteIndex)  # mute index for sum
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
