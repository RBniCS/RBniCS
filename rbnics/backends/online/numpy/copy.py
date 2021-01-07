# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic import copy as basic_copy
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping.function_copy import basic_function_copy
from rbnics.backends.online.numpy.wrapping.tensor_copy import basic_tensor_copy
from rbnics.utils.decorators import backend_for, list_of, ModuleWrapper

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping_for_wrapping = ModuleWrapper()
function_copy = basic_function_copy(backend, wrapping_for_wrapping)
tensor_copy = basic_tensor_copy(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(function_copy=function_copy, tensor_copy=tensor_copy)
copy_base = basic_copy(backend, wrapping)


@backend_for("numpy", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), ))
def copy(arg):
    return copy_base(arg)
