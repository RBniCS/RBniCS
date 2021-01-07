# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import ix_ as Slicer
from rbnics.backends.online.numpy.wrapping.basis_functions_matrix_mul import (
    basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector)
from rbnics.backends.online.numpy.wrapping.function_load import function_load
from rbnics.backends.online.numpy.wrapping.function_save import function_save
from rbnics.backends.online.numpy.wrapping.function_to_vector import function_to_vector
from rbnics.backends.online.numpy.wrapping.functions_list_mul import (
    functions_list_mul_online_matrix, functions_list_mul_online_vector)
from rbnics.backends.online.numpy.wrapping.get_mpi_comm import get_mpi_comm
from rbnics.backends.online.numpy.wrapping.gram_schmidt_projection_step import gram_schmidt_projection_step
from rbnics.backends.online.numpy.wrapping.matrix_mul import (
    matrix_mul_vector, vectorized_matrix_inner_vectorized_matrix)
from rbnics.backends.online.numpy.wrapping.tensor_load import tensor_load
from rbnics.backends.online.numpy.wrapping.tensor_save import tensor_save
from rbnics.backends.online.numpy.wrapping.vector_mul import vector_mul_vector

__all__ = [
    "basis_functions_matrix_mul_online_matrix",
    "basis_functions_matrix_mul_online_vector",
    "function_load",
    "function_save",
    "function_to_vector",
    "functions_list_mul_online_matrix",
    "functions_list_mul_online_vector",
    "get_mpi_comm",
    "gram_schmidt_projection_step",
    "matrix_mul_vector",
    "Slicer",
    "tensor_load",
    "tensor_save",
    "vector_mul_vector",
    "vectorized_matrix_inner_vectorized_matrix"
]
