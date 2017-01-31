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

import RBniCS.backends # avoid circular imports when importing fenics backend
from RBniCS.backends.fenics.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs

def evaluate_basis_functions_matrix_at_dofs(input_basis_functions_matrix, dofs_list, reduced_V, reduced_dofs_list):
    components = input_basis_functions_matrix._components_name
    output_basis_functions_matrix = RBniCS.backends.fenics.BasisFunctionsMatrix(reduced_V)
    output_basis_functions_matrix.init(components)
    if len(components) > 1:
        for component in components:
            input_functions_list = input_basis_functions_matrix._components[component]
            for basis_function in input_functions_list:
                reduced_basis_function = evaluate_sparse_function_at_dofs(basis_function, dofs_list, reduced_V, reduced_dofs_list)
                output_basis_functions_matrix.enrich(reduced_basis_function, component=component)
    else:
        input_functions_list = input_basis_functions_matrix._components[components[0]]
        for basis_function in input_functions_list:
            reduced_basis_function = evaluate_sparse_function_at_dofs(basis_function, dofs_list, reduced_V, reduced_dofs_list)
            output_basis_functions_matrix.enrich(reduced_basis_function)
    return output_basis_functions_matrix
    
            
