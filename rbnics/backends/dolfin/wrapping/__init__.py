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

from dolfin import log, CRITICAL, ERROR, WARNING, INFO, PROGRESS, TRACE, DEBUG # easier to read in parallel
from rbnics.backends.dolfin.wrapping.assemble import assemble
from rbnics.backends.dolfin.wrapping.assemble_operator_for_derivative import assemble_operator_for_derivative
from rbnics.backends.dolfin.wrapping.assemble_operator_for_restriction import assemble_operator_for_restriction
from rbnics.backends.dolfin.wrapping.basis_functions_matrix_mul import basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector
from rbnics.backends.dolfin.wrapping.compute_theta_for_derivative import compute_theta_for_derivative
from rbnics.backends.dolfin.wrapping.compute_theta_for_restriction import compute_theta_for_restriction
from rbnics.backends.dolfin.wrapping.counterclockwise import counterclockwise
from rbnics.backends.dolfin.wrapping.create_submesh import convert_functionspace_to_submesh, convert_meshfunctions_to_submesh, create_submesh, map_functionspaces_between_mesh_and_submesh
from rbnics.backends.dolfin.wrapping.dirichlet_bc import DirichletBC
from rbnics.backends.dolfin.wrapping.dofs_parallel_io_helpers import build_dof_map_writer_mapping, build_dof_map_reader_mapping
from rbnics.backends.dolfin.wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs import evaluate_and_vectorize_sparse_matrix_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_basis_functions_matrix_at_dofs import evaluate_basis_functions_matrix_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.backends.dolfin.wrapping.evaluate_sparse_vector_at_dofs import evaluate_sparse_vector_at_dofs
from rbnics.backends.dolfin.wrapping.expression_description import expression_description
from rbnics.backends.dolfin.wrapping.expression_iterator import expression_iterator
from rbnics.backends.dolfin.wrapping.expression_name import expression_name
from rbnics.backends.dolfin.wrapping.expression_replace import expression_replace
import rbnics.backends.dolfin.wrapping.form_and
from rbnics.backends.dolfin.wrapping.form_argument_replace import form_argument_replace
from rbnics.backends.dolfin.wrapping.form_argument_space import form_argument_space
from rbnics.backends.dolfin.wrapping.form_description import form_description
from rbnics.backends.dolfin.wrapping.form_iterator import form_iterator
import rbnics.backends.dolfin.wrapping.form_mul
from rbnics.backends.dolfin.wrapping.form_name import form_name
from rbnics.backends.dolfin.wrapping.form_replace import form_replace
from rbnics.backends.dolfin.wrapping.function_copy import function_copy
from rbnics.backends.dolfin.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.dolfin.wrapping.function_from_subfunction_if_any import function_from_subfunction_if_any
from rbnics.backends.dolfin.wrapping.function_from_ufl_operators import function_from_ufl_operators
from rbnics.backends.dolfin.wrapping.function_load import function_load
from rbnics.backends.dolfin.wrapping.function_save import function_save
from rbnics.backends.dolfin.wrapping.function_space import FunctionSpace
from rbnics.backends.dolfin.wrapping.function_to_vector import function_to_vector
from rbnics.backends.dolfin.wrapping.functions_list_mul import functions_list_mul_online_matrix, functions_list_mul_online_vector
from rbnics.backends.dolfin.wrapping.get_auxiliary_problem_for_non_parametrized_function import get_auxiliary_problem_for_non_parametrized_function
from rbnics.backends.dolfin.wrapping.get_function_norm import get_function_norm
from rbnics.backends.dolfin.wrapping.get_function_space import get_function_space
from rbnics.backends.dolfin.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.dolfin.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.dolfin.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.dolfin.wrapping.is_parametrized import is_parametrized
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component import is_problem_solution_or_problem_solution_component
from rbnics.backends.dolfin.wrapping.is_problem_solution_or_problem_solution_component_type import is_problem_solution_or_problem_solution_component_type
from rbnics.backends.dolfin.wrapping.is_time_dependent import is_time_dependent
from rbnics.backends.dolfin.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.dolfin.wrapping.parametrized_constant import is_parametrized_constant, ParametrizedConstant, parametrized_constant_to_float
from rbnics.backends.dolfin.wrapping.parametrized_expression import ParametrizedExpression
from rbnics.backends.dolfin.wrapping.petsc_ts_integrator import PETScTSIntegrator
from rbnics.backends.dolfin.wrapping.plot import plot
from rbnics.backends.dolfin.wrapping.pull_back_to_reference_domain import PullBackFormsToReferenceDomain
from rbnics.backends.dolfin.wrapping.solution_identify_component import solution_identify_component
from rbnics.backends.dolfin.wrapping.solution_iterator import solution_iterator
from rbnics.backends.dolfin.wrapping.tensor_copy import tensor_copy
from rbnics.backends.dolfin.wrapping.ufl_lagrange_interpolation import assert_lagrange_1, get_global_dof_component, get_global_dof_coordinates, ufl_lagrange_interpolation
from rbnics.backends.dolfin.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'assemble',
    'assemble_operator_for_derivative',
    'assemble_operator_for_restriction',
    'assert_lagrange_1',
    'basis_functions_matrix_mul_online_matrix',
    'basis_functions_matrix_mul_online_vector',
    'build_dof_map_reader_mapping',
    'build_dof_map_writer_mapping',
    'compute_theta_for_derivative',
    'compute_theta_for_restriction',
    'counterclockwise',
    'convert_functionspace_to_submesh',
    'convert_meshfunctions_to_submesh',
    'create_submesh',
    'DirichletBC',
    'evaluate_and_vectorize_sparse_matrix_at_dofs',
    'evaluate_basis_functions_matrix_at_dofs',
    'evaluate_sparse_function_at_dofs',
    'evaluate_sparse_vector_at_dofs',
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
    'function_from_subfunction_if_any',
    'function_from_ufl_operators',
    'function_load',
    'function_save',
    'function_to_vector',
    'FunctionSpace',
    'functions_list_mul_online_matrix',
    'functions_list_mul_online_vector',
    'get_auxiliary_problem_for_non_parametrized_function',
    'get_function_norm',
    'get_function_space',
    'get_function_subspace',
    'get_global_dof_component',
    'get_global_dof_coordinates',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'is_parametrized',
    'is_parametrized_constant',
    'is_problem_solution_or_problem_solution_component',
    'is_problem_solution_or_problem_solution_component_type',
    'is_time_dependent',
    'map_functionspaces_between_mesh_and_submesh',
    'matrix_mul_vector',
    'ParametrizedConstant',
    'parametrized_constant_to_float',
    'ParametrizedExpression',
    'PETScTSIntegrator',
    'plot',
    'PullBackFormsToReferenceDomain',
    'solution_identify_component',
    'solution_iterator',
    'tensor_copy',
    'ufl_lagrange_interpolation',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]

__overridden__ = {
    'rbnics': [
        'assemble_operator_for_derivative',
        'assemble_operator_for_restriction',
        'compute_theta_for_derivative',
        'compute_theta_for_restriction',
        'ParametrizedExpression',
        'PullBackFormsToReferenceDomain'
    ],
    'rbnics.utils.mpi': [
        'log', 'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'PROGRESS', 'TRACE', 'DEBUG'
    ]
}
