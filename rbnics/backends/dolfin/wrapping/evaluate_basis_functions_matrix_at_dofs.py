# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.dolfin.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.utils.decorators import ModuleWrapper


def basic_evaluate_basis_functions_matrix_at_dofs(backend, wrapping):

    def _basic_evaluate_basis_functions_matrix_at_dofs(input_basis_functions_matrix, dofs_list,
                                                       output_basis_functions_matrix, reduced_dofs_list):
        components = output_basis_functions_matrix._components_name
        reduced_space = output_basis_functions_matrix.space
        if len(components) > 1:
            for component in components:
                input_functions_list = input_basis_functions_matrix._components[component]
                for basis_function in input_functions_list:
                    reduced_basis_function = wrapping.evaluate_sparse_function_at_dofs(
                        basis_function, dofs_list, reduced_space, reduced_dofs_list)
                    output_basis_functions_matrix.enrich(reduced_basis_function, component=component)
        else:
            input_functions_list = input_basis_functions_matrix._components[components[0]]
            for basis_function in input_functions_list:
                reduced_basis_function = wrapping.evaluate_sparse_function_at_dofs(
                    basis_function, dofs_list, reduced_space, reduced_dofs_list)
                output_basis_functions_matrix.enrich(reduced_basis_function)
        return output_basis_functions_matrix

    return _basic_evaluate_basis_functions_matrix_at_dofs


backend = ModuleWrapper()
wrapping = ModuleWrapper(evaluate_sparse_function_at_dofs)
evaluate_basis_functions_matrix_at_dofs = basic_evaluate_basis_functions_matrix_at_dofs(backend, wrapping)
