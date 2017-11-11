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

from dolfin import FunctionSpace
from rbnics.backends.dolfin.wrapping.function_copy import function_copy

def basis_functions_matrix_mul_online_matrix(basis_functions_matrix, online_matrix, BasisFunctionsMatrixType):
    V = basis_functions_matrix.V_or_Z
    assert isinstance(V, FunctionSpace)
    
    reference_function = function_copy(basis_functions_matrix._components[basis_functions_matrix._basis_component_index_to_component_name[0]][0])
    reference_function.vector().zero()
    output = BasisFunctionsMatrixType(V)
    assert isinstance(online_matrix.M, dict)
    j = 0
    for (_, col_component_name) in sorted(basis_functions_matrix._basis_component_index_to_component_name.items()):
        for _ in range(online_matrix.M[col_component_name]):
            assert len(online_matrix[:, j]) == sum(len(functions_list) for functions_list in basis_functions_matrix._components)
            output_j = function_copy(reference_function)
            i = 0
            for (_, row_component_name) in sorted(basis_functions_matrix._basis_component_index_to_component_name.items()):
                for fun_i in basis_functions_matrix._components[row_component_name]:
                    online_matrix_ij = float(online_matrix[i, j])
                    output_j.vector().add_local(fun_i.vector().get_local()*online_matrix_ij)
                    i += 1
            output_j.vector().apply("add")
            output.enrich(output_j)
            j += 1
    return output

def basis_functions_matrix_mul_online_vector(basis_functions_matrix, online_vector):
    output = function_copy(basis_functions_matrix._components[basis_functions_matrix._basis_component_index_to_component_name[0]][0])
    output.vector().zero()
    i = 0
    for (_, component_name) in sorted(basis_functions_matrix._basis_component_index_to_component_name.items()):
        for fun_i in basis_functions_matrix._components[component_name]:
            online_vector_i = float(online_vector[i])
            output.vector().add_local(fun_i.vector().get_local()*online_vector_i)
            i += 1
    output.vector().apply("add")
    return output
