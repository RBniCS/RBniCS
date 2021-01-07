# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# AbstractTensorsList is used in place of concrete TensorsList to avoid unsolvable circular dependency
from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import overload


def basic_tensors_list_mul_online_function(backend, wrapping):

    def _basic_tensors_list_mul_online_function(tensors_list, online_function):
        output = wrapping.tensor_copy(tensors_list._list[0])
        _multiply(tensors_list, online_function, output)
        return output

    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: OnlineFunction.Type(),
                  output: backend.Matrix.Type()):
        output.zero()
        for (i, matrix_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output += matrix_i * online_vector_i

    @overload
    def _multiply(tensors_list: AbstractTensorsList, online_function: OnlineFunction.Type(),
                  output: backend.Vector.Type()):
        output.zero()
        for (i, vector_i) in enumerate(tensors_list._list):
            online_vector_i = online_function.vector()[i]
            output.add_local(vector_i.get_local() * online_vector_i)
        output.apply("add")

    return _basic_tensors_list_mul_online_function

# No explicit instantiation for backend = rbnics.backends.dolfin to avoid
# circular dependencies. The concrete instatiation will be carried out in
# rbnics.backends.dolfin.tensors_list
