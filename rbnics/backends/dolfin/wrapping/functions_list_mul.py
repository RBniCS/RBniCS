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

from dolfin import Function, FunctionSpace

def functions_list_mul_online_matrix(functions_list, online_matrix, FunctionsListType):
    space = functions_list.space
    assert isinstance(space, FunctionSpace)
    
    output = FunctionsListType(space)
    assert isinstance(online_matrix.M, int)
    for j in range(online_matrix.M):
        assert len(online_matrix[:, j]) == len(functions_list)
        output_j = Function(space)
        for (i, fun_i) in enumerate(functions_list):
            output_j.vector().add_local(fun_i.vector().get_local()*online_matrix[i, j])
        output_j.vector().apply("add")
        output.enrich(output_j)
    return output

def functions_list_mul_online_vector(functions_list, online_vector):
    space = functions_list.space
    assert isinstance(space, FunctionSpace)
    
    output = Function(space)
    if len(functions_list) == 0:
        return output
    else:
        for (i, fun_i) in enumerate(functions_list):
            output.vector().add_local(fun_i.vector().get_local()*online_vector[i])
        output.vector().apply("add")
        return output
