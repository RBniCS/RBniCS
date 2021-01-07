# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import TensorsList as AbstractTensorsList
from rbnics.backends.basic import TensorsList as BasicTensorsList
from rbnics.backends.online.numpy.copy import tensor_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import get_mpi_comm, tensor_load, tensor_save
from rbnics.backends.online.numpy.wrapping.tensors_list_mul import basic_tensors_list_mul_online_function
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper()
tensors_list_mul_online_function = basic_tensors_list_mul_online_function(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(get_mpi_comm, tensor_load, tensor_save, tensor_copy=tensor_copy,
                         tensors_list_mul_online_function=tensors_list_mul_online_function)
online_backend = ModuleWrapper(OnlineFunction=Function)
online_wrapping = ModuleWrapper()
TensorsList_Base = BasicTensorsList(backend, wrapping, online_backend, online_wrapping)


@BackendFor("numpy", inputs=(AbstractTensorsList, ))
class TensorsList(TensorsList_Base):
    pass
