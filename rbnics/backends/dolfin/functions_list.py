# Copyright (C) 2015-2023 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numbers import Number
try:
    from ufl_legacy.core.operator import Operator
except ImportError:
    from ufl.core.operator import Operator
from dolfin import FunctionSpace
from rbnics.backends.basic import FunctionsList as BasicFunctionsList
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.wrapping import (function_extend_or_restrict, function_from_ufl_operators, function_load,
                                             function_save, function_to_vector, functions_list_mul_online_matrix,
                                             functions_list_mul_online_vector, get_function_subspace, get_mpi_comm)
from rbnics.backends.online import OnlineFunction, OnlineMatrix, OnlineVector
from rbnics.backends.online.wrapping import function_to_vector as online_function_to_online_vector
from rbnics.utils.decorators import BackendFor, dict_of, list_of, ModuleWrapper, overload


def AdditionalIsFunction(arg):
    return isinstance(arg, Operator)


def ConvertAdditionalFunctionTypes(arg):
    assert isinstance(arg, Operator)
    return function_from_ufl_operators(arg)


backend = ModuleWrapper(Function)
wrapping = ModuleWrapper(function_extend_or_restrict, function_load, function_save, function_to_vector,
                         functions_list_mul_online_matrix, functions_list_mul_online_vector,
                         get_function_subspace, get_mpi_comm)
online_backend = ModuleWrapper(OnlineFunction=OnlineFunction, OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper(online_function_to_online_vector)
FunctionsList_Base = BasicFunctionsList(backend, wrapping, online_backend, online_wrapping,
                                        AdditionalIsFunction, ConvertAdditionalFunctionTypes)


@BackendFor("dolfin", inputs=(FunctionSpace, (str, None)))
class FunctionsList(FunctionsList_Base):
    def __init__(self, V, component=None):
        FunctionsList_Base.__init__(self, V, component)

    @overload(Operator, (None, str, dict_of(str, str)), (None, list_of(Number)), bool)
    def _enrich(self, function, component, weight, copy):
        function = function_from_ufl_operators(function)
        FunctionsList_Base._enrich(self, function, component, weight, copy)

    @overload(int, Operator)
    def __setitem__(self, key, item):
        item = function_from_ufl_operators(item)
        FunctionsList_Base.__setitem__(self, key, item)
