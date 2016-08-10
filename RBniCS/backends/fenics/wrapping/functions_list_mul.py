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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from dolfin import FunctionSpace
import RBniCS.backends.fenics.functions_list as FEniCS_backend # break circular imports
frorm RBniCS.backends.fenics.wrapping.function_copy import function_copy
from RBniCS.backends.numpy.matrix import Matrix_Type as OnlineMatrix_Type
from RBniCS.backends.numpy.vector import Vector_Type as OnlineVector_Type
from RBniCS.backends.numpy.function import Function_Type  as OnlineFunction_Type

def function_list_mul_online_matrix(functions_list, online_matrix):
    V = functions_list.V_or_Z
    assert isinstance(V, FunctionSpace)
    assert isinstance(online_matrix, OnlineMatrix_Type)
    
    output = FEniCS_backend.FunctionsList(V)
    dim = online_matrix.shape[1]
    for j in range(dim):
        assert len(online_matrix[:, j]) == len(functions_list._list)
        output_j = function_copy(functions_list._list[0])
        output_j.vector().zero()
        for (i, fun_i) in enumerate(functions_list._list):
            output_j.vector().add_local(fun_i.vector().array()*online_matrix[i, j])
        output_j.vector().apply("add")
        output.enrich(output_j)
    return output

def function_list_mul_online_vector(functions_list, online_vector):
    assert isinstance(online_vector, OnlineVector_Type)
    
    output = function_copy(functions_list._list[0])
    output.vector().zero()
    for (i, fun_i) in enumerate(self._list):
        output.vector().add_local(fun_i.vector().array()*online_vector.item(i))
    output.vector().apply("add")
    return output
    
def function_list_mul_online_function(functions_list, online_function):
    assert isinstance(online_function, OnlineFunction_Type)
    
    return function_list_mul_online_vector(functions_list, online_function.vector())
    
