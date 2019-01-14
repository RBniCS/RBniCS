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

from rbnics.eim.utils.decorators.store_map_from_basis_functions_to_reduced_problem import add_to_map_from_basis_functions_to_reduced_problem, get_reduced_problem_from_basis_functions, StoreMapFromBasisFunctionsToReducedProblem
from rbnics.eim.utils.decorators.store_map_from_each_basis_function_to_component_and_index import add_to_map_from_basis_function_to_component_and_index, get_component_and_index_from_basis_function, StoreMapFromEachBasisFunctionToComponentAndIndex
from rbnics.eim.utils.decorators.store_map_from_parametrized_expression_to_problem import add_to_map_from_parametrized_expression_to_problem, get_problem_from_parametrized_expression, StoreMapFromParametrizedExpressionToProblem
from rbnics.eim.utils.decorators.store_map_from_parametrized_operators_to_problem import add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator, StoreMapFromParametrizedOperatorsToProblem
from rbnics.eim.utils.decorators.store_map_from_parametrized_operators_to_term_and_index import add_to_map_from_parametrized_operator_to_term_and_index, get_term_and_index_from_parametrized_operator, StoreMapFromParametrizedOperatorsToTermAndIndex
from rbnics.eim.utils.decorators.store_map_from_riesz_storage_to_reduced_problem import add_to_map_from_error_estimation_inner_product_to_reduced_problem, add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem, add_to_map_from_riesz_solve_inner_product_to_reduced_problem, add_to_map_from_riesz_solve_storage_to_reduced_problem, get_reduced_problem_from_error_estimation_inner_product, get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc, get_reduced_problem_from_riesz_solve_inner_product, get_reduced_problem_from_riesz_solve_storage, StoreMapFromRieszStorageToReducedProblem

__all__ = [
    'add_to_map_from_basis_function_to_component_and_index',
    'add_to_map_from_basis_functions_to_reduced_problem',
    'add_to_map_from_error_estimation_inner_product_to_reduced_problem',
    'add_to_map_from_parametrized_operator_to_problem',
    'add_to_map_from_parametrized_operator_to_term_and_index',
    'add_to_map_from_parametrized_expression_to_problem',
    'add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem',
    'add_to_map_from_riesz_solve_inner_product_to_reduced_problem',
    'add_to_map_from_riesz_solve_storage_to_reduced_problem',
    'get_component_and_index_from_basis_function',
    'get_problem_from_parametrized_operator',
    'get_problem_from_parametrized_expression',
    'get_reduced_problem_from_basis_functions',
    'get_reduced_problem_from_error_estimation_inner_product',
    'get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc',
    'get_reduced_problem_from_riesz_solve_inner_product',
    'get_reduced_problem_from_riesz_solve_storage',
    'get_term_and_index_from_parametrized_operator',
    'StoreMapFromBasisFunctionsToReducedProblem',
    'StoreMapFromEachBasisFunctionToComponentAndIndex',
    'StoreMapFromParametrizedExpressionToProblem',
    'StoreMapFromParametrizedOperatorsToProblem',
    'StoreMapFromParametrizedOperatorsToTermAndIndex',
    'StoreMapFromRieszStorageToReducedProblem'
]
