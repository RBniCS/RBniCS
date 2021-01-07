# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.eim.utils.decorators.define_symbolic_parameters import DefineSymbolicParameters
from rbnics.eim.utils.decorators.store_map_from_basis_functions_to_reduced_problem import (
    add_to_map_from_basis_functions_to_reduced_problem, get_reduced_problem_from_basis_functions,
    StoreMapFromBasisFunctionsToReducedProblem)
from rbnics.eim.utils.decorators.store_map_from_each_basis_function_to_component_and_index import (
    add_to_map_from_basis_function_to_component_and_index, get_component_and_index_from_basis_function,
    StoreMapFromEachBasisFunctionToComponentAndIndex)
from rbnics.eim.utils.decorators.store_map_from_parametrized_expression_to_problem import (
    add_to_map_from_parametrized_expression_to_problem, get_problem_from_parametrized_expression,
    StoreMapFromParametrizedExpressionToProblem)
from rbnics.eim.utils.decorators.store_map_from_parametrized_operators_to_problem import (
    add_to_map_from_parametrized_operator_to_problem, get_problem_from_parametrized_operator,
    StoreMapFromParametrizedOperatorsToProblem)
from rbnics.eim.utils.decorators.store_map_from_parametrized_operators_to_term_and_index import (
    add_to_map_from_parametrized_operator_to_term_and_index, get_term_and_index_from_parametrized_operator,
    StoreMapFromParametrizedOperatorsToTermAndIndex)
from rbnics.eim.utils.decorators.store_map_from_riesz_storage_to_reduced_problem import (
    add_to_map_from_error_estimation_inner_product_to_reduced_problem,
    add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem,
    add_to_map_from_riesz_solve_inner_product_to_reduced_problem,
    add_to_map_from_riesz_solve_storage_to_reduced_problem,
    get_reduced_problem_from_error_estimation_inner_product,
    get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc,
    get_reduced_problem_from_riesz_solve_inner_product,
    get_reduced_problem_from_riesz_solve_storage,
    StoreMapFromRieszStorageToReducedProblem)

__all__ = [
    "add_to_map_from_basis_function_to_component_and_index",
    "add_to_map_from_basis_functions_to_reduced_problem",
    "add_to_map_from_error_estimation_inner_product_to_reduced_problem",
    "add_to_map_from_parametrized_operator_to_problem",
    "add_to_map_from_parametrized_operator_to_term_and_index",
    "add_to_map_from_parametrized_expression_to_problem",
    "add_to_map_from_riesz_solve_homogeneous_dirichlet_bc_to_reduced_problem",
    "add_to_map_from_riesz_solve_inner_product_to_reduced_problem",
    "add_to_map_from_riesz_solve_storage_to_reduced_problem",
    "DefineSymbolicParameters",
    "get_component_and_index_from_basis_function",
    "get_problem_from_parametrized_operator",
    "get_problem_from_parametrized_expression",
    "get_reduced_problem_from_basis_functions",
    "get_reduced_problem_from_error_estimation_inner_product",
    "get_reduced_problem_from_riesz_solve_homogeneous_dirichlet_bc",
    "get_reduced_problem_from_riesz_solve_inner_product",
    "get_reduced_problem_from_riesz_solve_storage",
    "get_term_and_index_from_parametrized_operator",
    "StoreMapFromBasisFunctionsToReducedProblem",
    "StoreMapFromEachBasisFunctionToComponentAndIndex",
    "StoreMapFromParametrizedExpressionToProblem",
    "StoreMapFromParametrizedOperatorsToProblem",
    "StoreMapFromParametrizedOperatorsToTermAndIndex",
    "StoreMapFromRieszStorageToReducedProblem"
]
