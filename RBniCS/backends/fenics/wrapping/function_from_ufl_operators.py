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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from ufl.algebra import Division, Product, Sum
from ufl.constantvalue import ScalarValue
from ufl.tensors import as_tensor, ComponentTensor
from dolfin import Function, GenericVector

def function_from_ufl_operators(input_function):
    assert isinstance(input_function, (Function, Sum, Product, Division, ComponentTensor))
    if isinstance(input_function, Function):
        return input_function
    elif isinstance(input_function, Sum):
        return _function_from_ufl_sum(input_function.ufl_operands[0], input_function.ufl_operands[1])
    elif isinstance(input_function, Product):
        return _function_from_ufl_product(input_function.ufl_operands[0], input_function.ufl_operands[1])
    elif isinstance(input_function, Division):
        return _function_from_ufl_division(input_function.ufl_operands[0], input_function.ufl_operands[1])
    elif isinstance(input_function, ComponentTensor):
        expression = input_function.ufl_operands[0]
        indices = input_function.ufl_operands[1].indices()
        assert isinstance(expression, (Sum, Product, Division))
        if isinstance(expression, Sum):
            addend_1 = as_tensor(expression.ufl_operands[0], indices)
            addend_2 = as_tensor(expression.ufl_operands[1], indices)
            return _function_from_ufl_sum(addend_1, addend_2)
        elif isinstance(expression, Product):
            factor_1 = expression.ufl_operands[0]
            factor_2 = expression.ufl_operands[1]
            assert isinstance(factor_1, (float, ScalarValue)) or isinstance(factor_2, (float, ScalarValue))
            if isinstance(factor_1, (float, ScalarValue)):
                factor_2 = as_tensor(factor_2, indices)
            else: # isinstance(factor_2, (float, ScalarValue))
                factor_1 = as_tensor(factor_1, indices)
            return _function_from_ufl_product(factor_1, factor_2)
        elif isinstance(expression, Division):
            nominator_function = as_tensor(expression.ufl_operands[0], indices)
            denominator = expression.ufl_operands[1]
            return _function_from_ufl_division(nominator_function, denominator)
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to function_from_ufl_operators")

def _function_from_ufl_sum(addend_1, addend_2):
    addend_1 = function_from_ufl_operators(addend_1)
    addend_2 = function_from_ufl_operators(addend_2)
    assert isinstance(addend_1, Function)
    assert isinstance(addend_2, Function)
    sum_ = addend_1.copy(deepcopy=True)
    sum_.vector().add_local(addend_2.vector().array())
    sum_.vector().apply("")
    return sum_
    
def _function_from_ufl_product(factor_1, factor_2):
    assert isinstance(factor_1, (float, ScalarValue)) or isinstance(factor_2, (float, ScalarValue))
    if isinstance(factor_1, (float, ScalarValue)):
        factor_2 = function_from_ufl_operators(factor_2)
        product = factor_2.copy(deepcopy=True)
        product.vector()[:] *= float(factor_1)
        return product
    else: # isinstance(factor_2, (float, ScalarValue))
        factor_1 = function_from_ufl_operators(factor_1)
        product = factor_1.copy(deepcopy=True)
        product.vector()[:] *= float(factor_2)
        return product
        
def _function_from_ufl_division(nominator, denominator):
    nominator = function_from_ufl_operators(nominator)
    assert isinstance(denominator, (float, ScalarValue))
    division = nominator.copy(deepcopy=True)
    division.vector()[:] /= float(denominator)
    return division
    
