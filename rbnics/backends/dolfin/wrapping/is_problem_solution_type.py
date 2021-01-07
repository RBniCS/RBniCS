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
from dolfin import Constant, Function
from dolfin.function.expression import BaseExpression
from rbnics.utils.decorators import overload


@overload
def is_problem_solution_type(node: (Argument, BaseExpression, Constant, ConstantValue, GeometricQuantity,
                                    IndexBase, MultiIndex, Operator)):
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
