# Copyright (C) 2015-2016 by the RBniCS authors
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
from dolfin import Function, GenericVector
from RBniCS.backends.fenics.wrapping import function_copy

def function_from_ufl_operators(input_function):
    assert isinstance(input_function, (Function, Sum, Product, Division))
    if isinstance(input_function, Function):
        return input_function
    elif isinstance(input_function, Sum):
        addend_1 = function_from_ufl_operators(input_function.ufl_operands[0])
        addend_2 = function_from_ufl_operators(input_function.ufl_operands[1])
        assert isinstance(addend_1, Function)
        assert isinstance(addend_2, Function)
        sum_ = function_copy(addend_1)
        sum_.vector().add_local(addend_2.vector().array())
        sum_.vector().apply("")
        return sum_
    elif isinstance(input_function, Product):
        factor_1 = input_function.ufl_operands[0]
        factor_2 = input_function.ufl_operands[1]
        assert isinstance(factor_1, (float, ScalarValue)) or isinstance(factor_2, (float, ScalarValue))
        if isinstance(factor_1, (float, ScalarValue)):
            assert isinstance(factor_2, Function)
            factor_2.vector()[:] *= float(factor_1)
            return factor_2
        else: # isinstance(factor_2, (float, ScalarValue))
            assert isinstance(factor_1, Function)
            factor_1.vector()[:] *= float(factor_2)
            return factor_1
    elif isinstance(input_function, Division):
        nominator = input_function.ufl_operands[0]
        denominator = input_function.ufl_operands[1]
        assert isinstance(nominator, Function)
        assert isinstance(denominator, (float, ScalarValue))
        nominator.vector()[:] /= float(denominator)
        return nominator
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to function_from_ufl_operators")
        
