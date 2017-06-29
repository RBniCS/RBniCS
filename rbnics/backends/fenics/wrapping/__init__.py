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

from rbnics.backends.fenics.wrapping.create_submesh import convert_functionspace_to_submesh, convert_meshfunctions_to_submesh, create_submesh
from rbnics.backends.fenics.wrapping.dirichlet_bc import DirichletBC
from rbnics.backends.fenics.wrapping.dofs_parallel_io_helpers import build_dof_map_writer_mapping, build_dof_map_reader_mapping
from rbnics.backends.fenics.wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs import evaluate_and_vectorize_sparse_matrix_at_dofs
from rbnics.backends.fenics.wrapping.evaluate_basis_functions_matrix_at_dofs import evaluate_basis_functions_matrix_at_dofs
from rbnics.backends.fenics.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.backends.fenics.wrapping.evaluate_sparse_vector_at_dofs import evaluate_sparse_vector_at_dofs
from rbnics.backends.fenics.wrapping.expression_on_reduced_mesh import expression_on_reduced_mesh
from rbnics.backends.fenics.wrapping.expression_on_truth_mesh import expression_on_truth_mesh
from rbnics.backends.fenics.wrapping.form_on_reduced_function_space import form_on_reduced_function_space
from rbnics.backends.fenics.wrapping.form_on_truth_function_space import form_on_truth_function_space
from rbnics.backends.fenics.wrapping.function_copy import function_copy
from rbnics.backends.fenics.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.fenics.wrapping.function_from_subfunction_if_any import function_from_subfunction_if_any
from rbnics.backends.fenics.wrapping.function_from_ufl_operators import function_from_ufl_operators
from rbnics.backends.fenics.wrapping.function_load import function_load
from rbnics.backends.fenics.wrapping.function_save import function_save
from rbnics.backends.fenics.wrapping.function_space import FunctionSpace
from rbnics.backends.fenics.wrapping.functions_list_basis_functions_matrix_mul import functions_list_basis_functions_matrix_mul_online_matrix, functions_list_basis_functions_matrix_mul_online_vector, functions_list_basis_functions_matrix_mul_online_function
from rbnics.backends.fenics.wrapping.get_auxiliary_problem_for_non_parametrized_function import get_auxiliary_problem_for_non_parametrized_function
from rbnics.backends.fenics.wrapping.get_expression_description import get_expression_description
from rbnics.backends.fenics.wrapping.get_expression_name import get_expression_name
from rbnics.backends.fenics.wrapping.get_form_argument import get_form_argument
from rbnics.backends.fenics.wrapping.get_form_description import get_form_description
from rbnics.backends.fenics.wrapping.get_form_name import get_form_name
from rbnics.backends.fenics.wrapping.get_function_space import get_function_space
from rbnics.backends.fenics.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.fenics.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.fenics.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.fenics.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.fenics.wrapping.parametrized_constant import is_parametrized_constant, ParametrizedConstant, parametrized_constant_to_float
from rbnics.backends.fenics.wrapping.parametrized_expression import ParametrizedExpression
from rbnics.backends.fenics.wrapping.plot import plot
from rbnics.backends.fenics.wrapping.tensor_copy import tensor_copy
from rbnics.backends.fenics.wrapping.tensor_load import tensor_load
from rbnics.backends.fenics.wrapping.tensor_save import tensor_save
from rbnics.backends.fenics.wrapping.tensors_list_mul import tensors_list_mul_online_function
from rbnics.backends.fenics.wrapping.ufl_lagrange_interpolation import assert_lagrange_1, get_global_dof_coordinates, get_global_dof_component, ufl_lagrange_interpolation
from rbnics.backends.fenics.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'assert_lagrange_1',
    'build_dof_map_reader_mapping',
    'build_dof_map_writer_mapping',
    'convert_functionspace_to_submesh',
    'convert_meshfunctions_to_submesh',
    'create_submesh',
    'DirichletBC',
    'evaluate_and_vectorize_sparse_matrix_at_dofs',
    'evaluate_basis_functions_matrix_at_dofs',
    'evaluate_sparse_function_at_dofs',
    'evaluate_sparse_vector_at_dofs',
    'expression_on_reduced_mesh',
    'expression_on_truth_mesh',
    'form_on_reduced_function_space',
    'form_on_truth_function_space',
    'function_copy',
    'function_extend_or_restrict',
    'function_from_subfunction_if_any',
    'function_from_ufl_operators',
    'function_load',
    'function_save',
    'FunctionSpace',
    'functions_list_basis_functions_matrix_mul_online_function',
    'functions_list_basis_functions_matrix_mul_online_matrix',
    'functions_list_basis_functions_matrix_mul_online_vector',
    'get_auxiliary_problem_for_non_parametrized_function',
    'get_expression_description',
    'get_expression_name',
    'get_form_argument',
    'get_form_description',
    'get_form_name',
    'get_function_space',
    'get_function_subspace',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'is_parametrized_constant',
    'matrix_mul_vector',
    'ParametrizedConstant',
    'parametrized_constant_to_float',
    'ParametrizedExpression',
    'plot',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'ufl_lagrange_interpolation',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]

__overridden__ = [
    'DirichletBC',
    'FunctionSpace',
    'ParametrizedExpression',
    'plot'
]
