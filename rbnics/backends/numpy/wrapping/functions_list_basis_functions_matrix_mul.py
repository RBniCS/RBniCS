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
## @file functions_list_basis_functions_matrix.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.abstract import FunctionsList as AbstractFunctionsList
from RBniCS.backends.basic.wrapping import functions_list_basis_functions_matrix_adapter
from RBniCS.backends.numpy.wrapping.function_copy import function_copy
import RBniCS.backends # avoid circular imports when importing numpy backend

def functions_list_basis_functions_matrix_mul_online_matrix(functions_list_basis_functions_matrix, online_matrix, FunctionsListType, backend):
    Z = functions_list_basis_functions_matrix.V_or_Z
    (functions, _) = functions_list_basis_functions_matrix_adapter(functions_list_basis_functions_matrix, backend)
    assert isinstance(Z, AbstractFunctionsList)
    assert isinstance(online_matrix, RBniCS.backends.numpy.Matrix.Type())
    
    output = FunctionsListType(Z)
    dim = online_matrix.shape[1]
    for j in range(dim):
        assert len(online_matrix[:, j]) == len(functions)
        output_j = function_copy(functions[0])
        output_j.vector()[:] = 0.
        for (i, fun_i) in enumerate(functions):
            output_j.vector()[:] += fun_i.vector()*online_matrix.item((i, j))
        output.enrich(output_j)
    return output

def functions_list_basis_functions_matrix_mul_online_vector(functions_list_basis_functions_matrix, online_vector, backend):
    (functions, _) = functions_list_basis_functions_matrix_adapter(functions_list_basis_functions_matrix, backend)
    assert isinstance(online_vector, (RBniCS.backends.numpy.Vector.Type(), tuple))
    
    output = function_copy(functions[0])
    output.vector()[:] = 0.
    if isinstance(online_vector, RBniCS.backends.numpy.Vector.Type()):
        for (i, fun_i) in enumerate(functions):
            output.vector()[:] += fun_i.vector()*online_vector.item(i)
    elif isinstance(online_vector, tuple):
        for (i, fun_i) in enumerate(functions):
            output.vector()[:] += fun_i.vector()*online_vector[i]
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in functions_list_basis_functions_matrix_mul_online_vector.")
        
    return output
    
def functions_list_basis_functions_matrix_mul_online_function(functions_list_basis_functions_matrix, online_function, backend):
    assert isinstance(online_function, RBniCS.backends.numpy.Function.Type())
    
    return functions_list_basis_functions_matrix_mul_online_vector(functions_list_basis_functions_matrix, online_function.vector(), backend)
    