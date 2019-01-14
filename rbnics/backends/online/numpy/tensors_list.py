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
wrapping = ModuleWrapper(get_mpi_comm, tensor_load, tensor_save, tensor_copy=tensor_copy, tensors_list_mul_online_function=tensors_list_mul_online_function)
online_backend = ModuleWrapper(OnlineFunction=Function)
online_wrapping = ModuleWrapper()
TensorsList_Base = BasicTensorsList(backend, wrapping, online_backend, online_wrapping)

@BackendFor("numpy", inputs=(AbstractTensorsList, ))
class TensorsList(TensorsList_Base):
    pass
