# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.basic import copy as basic_copy
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_copy, tensor_copy
from rbnics.utils.decorators import backend_for, list_of, ModuleWrapper

backend = ModuleWrapper(Function, Matrix, Vector)
wrapping = ModuleWrapper(function_copy, tensor_copy)
copy_base = basic_copy(backend, wrapping)


@backend_for("dolfin", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), ))
def copy(arg):
    return copy_base(arg)
