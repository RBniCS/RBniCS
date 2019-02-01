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

def basis_functions_matrix_mul_online_matrix(basis_functions_matrix, online_matrix, BasisFunctionsMatrixType):
    space = basis_functions_matrix.space
    assert isinstance(space, FunctionSpace)
    
    output = BasisFunctionsMatrixType(space)
    assert isinstance(online_matrix.M, dict)
    j = 0
    for col_component_name in basis_functions_matrix._components_name:
        for _ in range(online_matrix.M[col_component_name]):
            assert len(online_matrix[:, j]) == sum(len(functions_list) for functions_list in basis_functions_matrix._components)
            output_j = Function(space)
            i = 0
            for row_component_name in basis_functions_matrix._components_name:
                for fun_i in basis_functions_matrix._components[row_component_name]:
                    output_j.vector().add_local(fun_i.vector().get_local()*online_matrix[i, j])
                    i += 1
            output_j.vector().apply("add")
            output.enrich(output_j)
            j += 1
    return output

def basis_functions_matrix_mul_online_vector(basis_functions_matrix, online_vector):
    space = basis_functions_matrix.space
    assert isinstance(space, FunctionSpace)
    
    output = Function(space)
    if sum(basis_functions_matrix._component_name_to_basis_component_length.values()) == 0:
        return output
    else:
        i = 0
        for component_name in basis_functions_matrix._components_name:
            for fun_i in basis_functions_matrix._components[component_name]:
                output.vector().add_local(fun_i.vector().get_local()*online_vector[i])
                i += 1
        output.vector().apply("add")
        return output
