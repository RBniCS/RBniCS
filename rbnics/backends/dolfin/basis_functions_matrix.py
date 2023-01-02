# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import FunctionSpace
from rbnics.backends.basic import BasisFunctionsMatrix as BasicBasisFunctionsMatrix
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.wrapping import (basis_functions_matrix_mul_online_matrix,
                                             basis_functions_matrix_mul_online_vector,
                                             function_to_vector, get_function_subspace, get_mpi_comm)
from rbnics.backends.online import OnlineFunction, OnlineMatrix, OnlineVector
from rbnics.backends.online.wrapping import function_to_vector as online_function_to_online_vector
from rbnics.utils.decorators import BackendFor, list_of, ModuleWrapper

backend = ModuleWrapper(Function, FunctionsList)
wrapping = ModuleWrapper(basis_functions_matrix_mul_online_matrix, basis_functions_matrix_mul_online_vector,
                         function_to_vector, get_function_subspace, get_mpi_comm)
online_backend = ModuleWrapper(OnlineFunction=OnlineFunction, OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper(online_function_to_online_vector)
BasisFunctionsMatrix_Base = BasicBasisFunctionsMatrix(backend, wrapping, online_backend, online_wrapping)


@BackendFor("dolfin", inputs=(FunctionSpace, (list_of(str), str, None)))
class BasisFunctionsMatrix(BasisFunctionsMatrix_Base):
    pass
