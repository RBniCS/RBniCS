# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import FunctionsList as BasicFunctionsList
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import (function_load, function_save, function_to_vector,
                                                   functions_list_mul_online_matrix, functions_list_mul_online_vector,
                                                   get_mpi_comm)
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Function)
wrapping = ModuleWrapper(function_load, function_save, function_to_vector, functions_list_mul_online_matrix,
                         functions_list_mul_online_vector, get_mpi_comm)
online_backend = ModuleWrapper(OnlineFunction=Function, OnlineMatrix=Matrix, OnlineVector=Vector)
online_wrapping = ModuleWrapper(function_to_vector)
FunctionsList_Base = BasicFunctionsList(backend, wrapping, online_backend, online_wrapping)


@BackendFor("numpy", inputs=(AbstractFunctionsList, (str, None)))
class FunctionsList(FunctionsList_Base):
    def __init__(self, basis_functions, component=None):
        FunctionsList_Base.__init__(self, basis_functions, component)
