# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.basic import TensorsList as BasicTensorsList
from rbnics.backends.dolfin.export import tensor_save
from rbnics.backends.dolfin.import_ import tensor_load
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import get_mpi_comm, tensor_copy
from rbnics.backends.dolfin.wrapping.tensors_list_mul import basic_tensors_list_mul_online_function
from rbnics.backends.online import OnlineFunction
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper(tensor_copy)
tensors_list_mul_online_function = basic_tensors_list_mul_online_function(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(get_mpi_comm, tensor_copy, tensor_load=tensor_load, tensor_save=tensor_save,
                         tensors_list_mul_online_function=tensors_list_mul_online_function)
online_backend = ModuleWrapper(OnlineFunction=OnlineFunction)
online_wrapping = ModuleWrapper()
TensorsList_Base = BasicTensorsList(backend, wrapping, online_backend, online_wrapping)


@BackendFor("dolfin", inputs=(FunctionSpace, ))
class TensorsList(TensorsList_Base):
    pass
