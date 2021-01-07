# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic import AffineExpansionStorage as BasicAffineExpansionStorage
from rbnics.backends.online.numpy.copy import function_copy, tensor_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import function_load, function_save, tensor_load, tensor_save
from rbnics.utils.decorators import BackendFor, ModuleWrapper, tuple_of

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(function_load, function_save, tensor_load, tensor_save, function_copy=function_copy,
                         tensor_copy=tensor_copy)
AffineExpansionStorage_Base = BasicAffineExpansionStorage(backend, wrapping)


@BackendFor("numpy", inputs=((int, tuple_of(Matrix.Type()), tuple_of(Vector.Type())), (int, None)))
class AffineExpansionStorage(AffineExpansionStorage_Base):
    def __init__(self, arg1, arg2=None):
        AffineExpansionStorage_Base.__init__(self, arg1, arg2)
