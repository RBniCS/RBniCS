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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import FunctionsList as BasicFunctionsList
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_load, function_save, function_to_vector, functions_list_mul_online_matrix, functions_list_mul_online_vector, get_mpi_comm
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Function)
wrapping = ModuleWrapper(function_load, function_save, function_to_vector, functions_list_mul_online_matrix, functions_list_mul_online_vector, get_mpi_comm)
online_backend = ModuleWrapper(OnlineFunction=Function, OnlineMatrix=Matrix, OnlineVector=Vector)
online_wrapping = ModuleWrapper(function_to_vector)
FunctionsList_Base = BasicFunctionsList(backend, wrapping, online_backend, online_wrapping)

@BackendFor("numpy", inputs=(AbstractFunctionsList, (str, None)))
class FunctionsList(FunctionsList_Base):
    def __init__(self, basis_functions, component=None):
        FunctionsList_Base.__init__(self, basis_functions, component)
