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

from rbnics.backends.dolfin.wrapping.assemble import assemble
from rbnics.backends.dolfin.wrapping.assemble_operator_for_derivative import assemble_operator_for_derivative
from rbnics.backends.dolfin.wrapping.assemble_operator_for_derivatives import assemble_operator_for_derivatives
from rbnics.backends.dolfin.wrapping.assemble_operator_for_restriction import assemble_operator_for_restriction
from rbnics.backends.dolfin.wrapping.assemble_operator_for_stability_factor import assemble_operator_for_stability_factor
from rbnics.backends.dolfin.wrapping.assemble_operator_for_supremizers import assemble_operator_for_supremizers
from rbnics.backends.dolfin.wrapping.basis_functions_matrix_mul import basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector
from rbnics.backends.dolfin.wrapping.compute_theta_for_derivative import compute_theta_for_derivative
from rbnics.backends.dolfin.wrapping.compute_theta_for_derivatives import compute_theta_for_derivatives
from rbnics.backends.dolfin.wrapping.compute_theta_for_restriction import compute_theta_for_restriction
from rbnics.backends.dolfin.wrapping.compute_theta_for_stability_factor import compute_theta_for_stability_factor
from rbnics.backends.dolfin.wrapping.compute_theta_for_supremizers import compute_theta_for_supremizers
from rbnics.backends.dolfin.wrapping.counterclockwise import counterclockwise
from rbnics.backends.dolfin.wrapping.create_submesh import convert_functionspace_to_submesh, convert_meshfunctions_to_submesh, create_submesh, map_functionspaces_between_mesh_and_submesh
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC
from rbnics.backends.dolfin.wrapping.dofs_parallel_io_helpers import build_dof_map_writer_mapping, build_dof_map_reader_mapping
from rbnics.backends.dolfin.wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs import evaluate_and_vectorize_sparse_matrix_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_basis_functions_matrix_at_dofs import evaluate_basis_functions_matrix_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_expression import evaluate_expression
from rbnics.backends.dolfin.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_sparse_vector_at_dofs import evaluate_sparse_vector_at_dofs
from rbnics.backends.dolfin.wrapping.expand_sum_product import expand_sum_product
from rbnics.backends.dolfin.wrapping.expression_description import expression_description
from rbnics.backends.dolfin.wrapping.expression_iterator import expression_iterator
from rbnics.backends.dolfin.wrapping.expression_name import expression_name
from rbnics.backends.dolfin.wrapping.expression_replace import expression_replace
from rbnics.backends.dolfin.wrapping.form_argument_replace import form_argument_replace
from rbnics.backends.dolfin.wrapping.form_argument_space import form_argument_space
from rbnics.backends.dolfin.wrapping.form_description import form_description
from rbnics.backends.dolfin.wrapping.form_iterator import form_iterator
import rbnics.backends.dolfin.wrapping.form_mul  # noqa: F401
from rbnics.backends.dolfin.wrapping.form_name import form_name
from rbnics.backends.dolfin.wrapping.form_replace import form_replace
from rbnics.backends.dolfin.wrapping.function_copy import function_copy
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.function_from_ufl_operators import function_from_ufl_operators
from rbnics.backends.dolfin.wrapping.function_load import function_load
from rbnics.backends.dolfin.wrapping.function_save import function_save
from rbnics.backends.dolfin.wrapping.function_space import FunctionSpace
from rbnics.backends.dolfin.wrapping.functions_list_mul import functions_list_mul_online_matrix, functions_list_mul_online_vector
from rbnics.backends.dolfin.wrapping.function_to_vector import function_to_vector
from rbnics.backends.dolfin.wrapping.generate_function_space_for_stability_factor import generate_function_space_for_stability_factor
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import get_auxiliary_problem_for_non_parametrized_function
from rbnics.backends.dolfin.wrapping.get_default_linear_solver import get_default_linear_solver
from rbnics.backends.dolfin.wrapping.get_function_norm import get_function_norm
from rbnics.backends.dolfin.wrapping.get_function_space import get_function_space
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.dolfin.wrapping.get_global_dof_component import get_global_dof_component
from rbnics.backends.dolfin.wrapping.get_global_dof_coordinates import get_global_dof_coordinates
from rbnics.backends.dolfin.wrapping.get_global_dof_to_local_dof_map import get_global_dof_to_local_dof_map
from rbnics.backends.dolfin.wrapping.get_local_dof_to_component_map import get_local_dof_to_component_map
from rbnics.backends.dolfin.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.dolfin.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.dolfin.wrapping.is_parametrized import is_parametrized
from rbnics.backends.dolfin.wrapping.is_problem_solution import is_problem_solution
from rbnics.backends.dolfin.wrapping.is_problem_solution_dot import is_problem_solution_dot
from rbnics.backends.dolfin.wrapping.is_problem_solution_type import is_problem_solution_type
from rbnics.backends.dolfin.wrapping.is_time_dependent import is_time_dependent
from rbnics.backends.dolfin.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.dolfin.wrapping.parametrized_constant import is_parametrized_constant, ParametrizedConstant, parametrized_constant_to_float
from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression
from rbnics.backends.dolfin.wrapping.plot import plot
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import is_pull_back_expression, is_pull_back_expression_parametrized, PullBackFormsToReferenceDomain
from rbnics.backends.dolfin.wrapping.remove_complex_nodes import remove_complex_nodes
from rbnics.backends.dolfin.wrapping.rewrite_quotients import rewrite_quotients
from rbnics.backends.dolfin.wrapping.solution_dot_identify_component import solution_dot_identify_component
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.backends.dolfin.wrapping.tensor_copy import tensor_copy
from rbnics.backends.dolfin.wrapping.to_petsc4py import to_petsc4py
from rbnics.backends.dolfin.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'assemble',
    'assemble_operator_for_derivative',
    'assemble_operator_for_derivatives',
    'assemble_operator_for_restriction',
    'assemble_operator_for_stability_factor',
    'assemble_operator_for_supremizers',
    'basis_functions_matrix_mul_online_matrix',
    'basis_functions_matrix_mul_online_vector',
    'build_dof_map_reader_mapping',
    'build_dof_map_writer_mapping',
    'compute_theta_for_derivative',
    'compute_theta_for_derivatives',
    'compute_theta_for_restriction',
    'compute_theta_for_stability_factor',
    'compute_theta_for_supremizers',
    'counterclockwise',
    'convert_functionspace_to_submesh',
    'convert_meshfunctions_to_submesh',
    'create_submesh',
    'DirichletBC',
    'evaluate_and_vectorize_sparse_matrix_at_dofs',
    'evaluate_basis_functions_matrix_at_dofs',
    'evaluate_expression',
    'evaluate_sparse_function_at_dofs',
    'evaluate_sparse_vector_at_dofs',
    'expand_sum_product',
    'expression_description',
    'expression_iterator',
    'expression_name',
    'expression_replace',
    'form_argument_replace',
    'form_argument_space',
    'form_description',
    'form_iterator',
    'form_name',
    'form_replace',
    'function_copy',
    'function_extend_or_restrict',
    'function_from_ufl_operators',
    'function_load',
    'function_save',
    'functions_list_mul_online_matrix',
    'functions_list_mul_online_vector',
    'FunctionSpace',
    'function_to_vector',
    'generate_function_space_for_stability_factor',
    'get_auxiliary_problem_for_non_parametrized_function',
    'get_default_linear_solver',
    'get_function_norm',
    'get_function_space',
    'get_function_subspace',
    'get_global_dof_component',
    'get_global_dof_coordinates',
    'get_global_dof_to_local_dof_map',
    'get_local_dof_to_component_map',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'is_parametrized',
    'is_parametrized_constant',
    'is_problem_solution',
    'is_problem_solution_dot',
    'is_problem_solution_type',
    'is_pull_back_expression',
    'is_pull_back_expression_parametrized',
    'is_time_dependent',
    'map_functionspaces_between_mesh_and_submesh',
    'matrix_mul_vector',
    'ParametrizedConstant',
    'parametrized_constant_to_float',
    'ParametrizedExpression',
    'plot',
    'PullBackFormsToReferenceDomain',
    'remove_complex_nodes',
    'rewrite_quotients',
    'solution_dot_identify_component',
    'solution_identify_component',
    'solution_iterator',
    'tensor_copy',
    'to_petsc4py',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]

__overridden__ = {
    'rbnics': [
        'assemble_operator_for_derivatives',
        'assemble_operator_for_stability_factor',
        'assemble_operator_for_supremizers',
        'compute_theta_for_derivatives',
        'compute_theta_for_stability_factor',
        'compute_theta_for_supremizers',
        'generate_function_space_for_stability_factor',
        'ParametrizedExpression',
        'plot',
        'PullBackFormsToReferenceDomain'
    ],
}
