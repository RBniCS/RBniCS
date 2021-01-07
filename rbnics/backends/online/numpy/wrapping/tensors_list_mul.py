# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# AbstractTensorsList is used in place of concrete TensorsList to avoid unsolvable circular dependency
from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.utils.decorators import overload


def basic_tensors_list_mul_online_function(backend, wrapping):
    def _basic_tensors_list_mul_online_function(tensors_list, online_function):
        output = wrapping.tensor_copy(tensors_list._list[0])
        _multiply(tensors_list, online_function, output)
        return output

    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: backend.Function.Type(),
                  output: backend.Matrix.Type()):
        output[:, :] = 0.
        for (i, tensor_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output[:, :] += tensor_i * online_vector_i

    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: backend.Function.Type(),
                  output: backend.Vector.Type()):
        output[:] = 0.
        for (i, tensor_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output[:] += tensor_i * online_vector_i

    return _basic_tensors_list_mul_online_function

# No explicit instantiation for backend = rbnics.backends.online.numpy to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.online.numpy.tensors_list
