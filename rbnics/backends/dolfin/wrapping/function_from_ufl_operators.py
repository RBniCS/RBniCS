# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
from ufl.algebra import Division, Product, Sum
from ufl.constantvalue import ScalarValue
from ufl.core.multiindex import IndexBase
from ufl.core.operator import Operator
from ufl.tensors import as_tensor, ComponentTensor
from dolfin import Function
from rbnics.backends.dolfin.wrapping.parametrized_constant import (
    is_parametrized_constant, parametrized_constant_to_float)
from rbnics.utils.decorators import overload, tuple_of


# Function from Function: do nothing
@overload
def function_from_ufl_operators(input_function: Function):
    return input_function


# Function from Sum
@overload
def function_from_ufl_operators(input_function: Sum):
    return _function_from_ufl_sum(input_function.ufl_operands[0], input_function.ufl_operands[1])


def _function_from_ufl_sum(addend_1, addend_2):
    addend_1 = function_from_ufl_operators(addend_1)
    addend_2 = function_from_ufl_operators(addend_2)
    assert isinstance(addend_1, Function)
    assert isinstance(addend_2, Function)
    sum_ = addend_1.copy(deepcopy=True)
    sum_.vector().add_local(addend_2.vector().get_local())
    sum_.vector().apply("")
    return sum_


# Function from Product
@overload
def function_from_ufl_operators(input_function: Product):
    return _function_from_ufl_product(input_function.ufl_operands[0], input_function.ufl_operands[1])


@overload
def _function_from_ufl_product(factor_1: (Number, ScalarValue), factor_2: (Function, Operator)):
    factor_2 = function_from_ufl_operators(factor_2)
    product = factor_2.copy(deepcopy=True)
    product.vector()[:] *= float(factor_1)
    return product


@overload
def _function_from_ufl_product(factor_1: (Function, Operator), factor_2: (Number, ScalarValue)):
    factor_1 = function_from_ufl_operators(factor_1)
    product = factor_1.copy(deepcopy=True)
    product.vector()[:] *= float(factor_2)
    return product


# Function from Division
@overload
def function_from_ufl_operators(input_function: Division):
    return _function_from_ufl_division(input_function.ufl_operands[0], input_function.ufl_operands[1])


def _function_from_ufl_division(nominator, denominator):
    nominator = function_from_ufl_operators(nominator)
    if is_parametrized_constant(denominator):
        denominator = parametrized_constant_to_float(denominator)
    assert isinstance(denominator, (Number, ScalarValue))
    division = nominator.copy(deepcopy=True)
    division.vector()[:] /= float(denominator)
    return division


# Function from ComponentTensor
@overload
def function_from_ufl_operators(input_function: ComponentTensor):
    return _function_from_ufl_component_tensor(input_function.ufl_operands[0], input_function.ufl_operands[1].indices())


@overload
def _function_from_ufl_component_tensor(expression: Sum, indices: tuple_of(IndexBase)):
    addend_1 = as_tensor(expression.ufl_operands[0], indices)
    addend_2 = as_tensor(expression.ufl_operands[1], indices)
    return _function_from_ufl_sum(addend_1, addend_2)


@overload
def _function_from_ufl_component_tensor(expression: Product, indices: tuple_of(IndexBase)):
    factor_1 = expression.ufl_operands[0]
    factor_2 = expression.ufl_operands[1]
    assert isinstance(factor_1, (Number, ScalarValue)) or isinstance(factor_2, (Number, ScalarValue))
    if isinstance(factor_1, (Number, ScalarValue)):
        factor_2 = as_tensor(factor_2, indices)
    else:  # isinstance(factor_2, (Number, ScalarValue))
        factor_1 = as_tensor(factor_1, indices)
    return _function_from_ufl_product(factor_1, factor_2)


@overload
def _function_from_ufl_component_tensor(expression: Division, indices: tuple_of(IndexBase)):
    nominator_function = as_tensor(expression.ufl_operands[0], indices)
    denominator = expression.ufl_operands[1]
    return _function_from_ufl_division(nominator_function, denominator)
