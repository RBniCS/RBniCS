# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList
from rbnics.backends.basic import BasisFunctionsMatrix as BasicBasisFunctionsMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.functions_list import FunctionsList
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import (basis_functions_matrix_mul_online_matrix,
                                                   basis_functions_matrix_mul_online_vector,
                                                   function_to_vector, get_mpi_comm)
from rbnics.utils.decorators import BackendFor, ModuleWrapper

backend = ModuleWrapper(Function, FunctionsList)
wrapping = ModuleWrapper(basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector,
                         function_to_vector, get_mpi_comm)
online_backend = ModuleWrapper(OnlineFunction=Function, OnlineMatrix=Matrix, OnlineVector=Vector)
online_wrapping = ModuleWrapper(function_to_vector)
BasisFunctionsMatrix_Base = BasicBasisFunctionsMatrix(backend, wrapping, online_backend, online_wrapping)


@BackendFor("numpy", inputs=(AbstractFunctionsList, ))
class BasisFunctionsMatrix(BasisFunctionsMatrix_Base):
    pass
