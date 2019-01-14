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

from rbnics.backends.online.basic import transpose as basic_transpose
from rbnics.backends.online.basic.wrapping import DelayedTransposeWithArithmetic as BasicDelayedTransposeWithArithmetic
from rbnics.backends.online.numpy.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.functions_list import FunctionsList
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.non_affine_expansion_storage import NonAffineExpansionStorage
from rbnics.backends.online.numpy.tensors_list import TensorsList
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_to_vector, matrix_mul_vector, vector_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.utils.decorators import backend_for, ModuleWrapper

backend = ModuleWrapper(BasisFunctionsMatrix, Function, FunctionsList, Matrix, NonAffineExpansionStorage, TensorsList, Vector)
DelayedTransposeWithArithmetic = BasicDelayedTransposeWithArithmetic(backend)
wrapping = ModuleWrapper(function_to_vector, matrix_mul_vector, vector_mul_vector, vectorized_matrix_inner_vectorized_matrix, DelayedTransposeWithArithmetic=DelayedTransposeWithArithmetic)
online_backend = ModuleWrapper(OnlineMatrix=Matrix, OnlineVector=Vector)
online_wrapping = ModuleWrapper()
transpose_base = basic_transpose(backend, wrapping, online_backend, online_wrapping)

@backend_for("numpy", inputs=((BasisFunctionsMatrix, DelayedTransposeWithArithmetic, Function.Type(), FunctionsList, TensorsList, Vector.Type()), ))
def transpose(arg):
    return transpose_base(arg)
