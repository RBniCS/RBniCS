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

from rbnics.backends.basic.wrapping.basis_functions_matrix_mul import basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector
from rbnics.backends.basic.wrapping.delayed_basis_functions_matrix import DelayedBasisFunctionsMatrix
from rbnics.backends.basic.wrapping.delayed_functions_list import DelayedFunctionsList
from rbnics.backends.basic.wrapping.delayed_linear_solver import DelayedLinearSolver
from rbnics.backends.basic.wrapping.delayed_product import DelayedProduct
from rbnics.backends.basic.wrapping.delayed_sum import DelayedSum
from rbnics.backends.basic.wrapping.delayed_transpose import DelayedTranspose
from rbnics.backends.basic.wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs import evaluate_and_vectorize_sparse_matrix_at_dofs
from rbnics.backends.basic.wrapping.evaluate_sparse_function_at_dofs import evaluate_sparse_function_at_dofs
from rbnics.backends.basic.wrapping.evaluate_sparse_vector_at_dofs import evaluate_sparse_vector_at_dofs
from rbnics.backends.basic.wrapping.expression_description import expression_description
from rbnics.backends.basic.wrapping.expression_iterator import expression_iterator
from rbnics.backends.basic.wrapping.expression_name import expression_name
from rbnics.backends.basic.wrapping.expression_on_reduced_mesh import expression_on_reduced_mesh
from rbnics.backends.basic.wrapping.expression_on_truth_mesh import expression_on_truth_mesh
from rbnics.backends.basic.wrapping.form_description import form_description
from rbnics.backends.basic.wrapping.form_iterator import form_iterator
from rbnics.backends.basic.wrapping.form_name import form_name
from rbnics.backends.basic.wrapping.form_on_reduced_function_space import form_on_reduced_function_space
from rbnics.backends.basic.wrapping.form_on_truth_function_space import form_on_truth_function_space
from rbnics.backends.basic.wrapping.function_copy import function_copy
from rbnics.backends.basic.wrapping.function_extend_or_restrict import function_extend_or_restrict
from rbnics.backends.basic.wrapping.function_load import function_load
from rbnics.backends.basic.wrapping.function_save import function_save
from rbnics.backends.basic.wrapping.functions_list_mul import functions_list_mul_online_matrix, functions_list_mul_online_vector
from rbnics.backends.basic.wrapping.get_function_space import get_function_space
from rbnics.backends.basic.wrapping.get_function_subspace import get_function_subspace
from rbnics.backends.basic.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.basic.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.basic.wrapping.is_parametrized import is_parametrized
from rbnics.backends.basic.wrapping.is_time_dependent import is_time_dependent
from rbnics.backends.basic.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.basic.wrapping.tensor_copy import tensor_copy
from rbnics.backends.basic.wrapping.tensor_load import tensor_load
from rbnics.backends.basic.wrapping.tensor_save import tensor_save
from rbnics.backends.basic.wrapping.tensors_list_mul import tensors_list_mul_online_function
from rbnics.backends.basic.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'basis_functions_matrix_mul_online_matrix',
    'basis_functions_matrix_mul_online_vector',
    'DelayedBasisFunctionsMatrix',
    'DelayedFunctionsList',
    'DelayedLinearSolver',
    'DelayedProduct',
    'DelayedSum',
    'DelayedTranspose',
    'evaluate_and_vectorize_sparse_matrix_at_dofs',
    'evaluate_sparse_function_at_dofs',
    'evaluate_sparse_vector_at_dofs',
    'expression_description',
    'expression_iterator',
    'expression_name',
    'expression_on_reduced_mesh',
    'expression_on_truth_mesh',
    'form_description',
    'form_iterator',
    'form_name',
    'form_on_reduced_function_space',
    'form_on_truth_function_space',
    'function_copy',
    'function_extend_or_restrict',
    'function_load',
    'function_save',
    'functions_list_mul_online_matrix',
    'functions_list_mul_online_vector',
    'get_function_space',
    'get_function_subspace',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'is_parametrized',
    'is_time_dependent',
    'matrix_mul_vector',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]
