# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic import transpose as basic_transpose
from rbnics.backends.online.basic.wrapping import DelayedTransposeWithArithmetic as BasicDelayedTransposeWithArithmetic
from rbnics.backends.online.numpy.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.functions_list import FunctionsList
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.non_affine_expansion_storage import NonAffineExpansionStorage
from rbnics.backends.online.numpy.tensors_list import TensorsList
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import (function_to_vector, matrix_mul_vector, vector_mul_vector,
                                                   vectorized_matrix_inner_vectorized_matrix)
from rbnics.utils.decorators import backend_for, ModuleWrapper

backend = ModuleWrapper(BasisFunctionsMatrix, Function, FunctionsList, Matrix, NonAffineExpansionStorage,
                        TensorsList, Vector)
DelayedTransposeWithArithmetic = BasicDelayedTransposeWithArithmetic(backend)
wrapping = ModuleWrapper(function_to_vector, matrix_mul_vector, vector_mul_vector,
                         vectorized_matrix_inner_vectorized_matrix,
                         DelayedTransposeWithArithmetic=DelayedTransposeWithArithmetic)
online_backend = ModuleWrapper(OnlineMatrix=Matrix, OnlineVector=Vector)
online_wrapping = ModuleWrapper()
transpose_base = basic_transpose(backend, wrapping, online_backend, online_wrapping)


@backend_for("numpy", inputs=((BasisFunctionsMatrix, DelayedTransposeWithArithmetic, Function.Type(),
                               FunctionsList, TensorsList, Vector.Type()), ))
def transpose(arg):
    return transpose_base(arg)
