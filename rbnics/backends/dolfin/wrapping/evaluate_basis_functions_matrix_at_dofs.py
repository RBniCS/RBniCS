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

def basic_evaluate_basis_functions_matrix_at_dofs(backend, wrapping):
    def _basic_evaluate_basis_functions_matrix_at_dofs(input_basis_functions_matrix, dofs_list, output_basis_functions_matrix, reduced_dofs_list):
        components = output_basis_functions_matrix._components_name
        reduced_space = output_basis_functions_matrix.space
        if len(components) > 1:
            for component in components:
                input_functions_list = input_basis_functions_matrix._components[component]
                for basis_function in input_functions_list:
                    reduced_basis_function = wrapping.evaluate_sparse_function_at_dofs(basis_function, dofs_list, reduced_space, reduced_dofs_list)
                    output_basis_functions_matrix.enrich(reduced_basis_function, component=component)
        else:
            input_functions_list = input_basis_functions_matrix._components[components[0]]
            for basis_function in input_functions_list:
                reduced_basis_function = wrapping.evaluate_sparse_function_at_dofs(basis_function, dofs_list, reduced_space, reduced_dofs_list)
                output_basis_functions_matrix.enrich(reduced_basis_function)
        return output_basis_functions_matrix
    return _basic_evaluate_basis_functions_matrix_at_dofs

from rbnics.backends.dolfin.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.utils.decorators import ModuleWrapper
backend = ModuleWrapper()
wrapping = ModuleWrapper(evaluate_sparse_function_at_dofs)
evaluate_basis_functions_matrix_at_dofs = basic_evaluate_basis_functions_matrix_at_dofs(backend, wrapping)
