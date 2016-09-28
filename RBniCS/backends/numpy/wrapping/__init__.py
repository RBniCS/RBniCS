# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file __init__.py
#  @brief Init file for auxiliary linear algebra module
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.numpy.wrapping.function_component import function_component
from RBniCS.backends.numpy.wrapping.function_copy import function_copy
from RBniCS.backends.numpy.wrapping.function_load import function_load
from RBniCS.backends.numpy.wrapping.function_save import function_save
from RBniCS.backends.numpy.wrapping.functions_list_mul import functions_list_mul_online_matrix, functions_list_mul_online_vector, functions_list_mul_online_function
from RBniCS.backends.numpy.wrapping.get_mpi_comm import get_mpi_comm
from RBniCS.backends.numpy.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from RBniCS.backends.numpy.wrapping.matrix_mul import matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix
from RBniCS.backends.basic.wrapping.tensor_copy import tensor_copy
from RBniCS.backends.basic.wrapping.tensor_load import tensor_load
from RBniCS.backends.basic.wrapping.tensor_save import tensor_save
from RBniCS.backends.basic.wrapping.tensors_list_mul import tensors_list_mul_online_function
from RBniCS.backends.numpy.wrapping.vector_mul import vector_mul_vector

__all__ = [
    'function_component',
    'function_copy',
    'function_load',
    'function_save',
    'functions_list_mul_online_matrix', 
    'functions_list_mul_online_vector', 
    'functions_list_mul_online_function',
    'get_mpi_comm',
    'gram_schmidt_projection_step',
    'matrix_mul_vector',
    'tensor_copy',
    'tensor_load',
    'tensor_save',
    'tensors_list_mul_online_function',
    'vector_mul_vector',
    'vectorized_matrix_inner_vectorized_matrix'
]
