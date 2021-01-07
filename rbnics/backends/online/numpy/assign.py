# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from rbnics.backends.online.basic.assign import assign as basic_assign
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, list_of, ModuleWrapper

backend = ModuleWrapper(Function, Matrix, Vector)
assign_base = basic_assign(backend)


@backend_for("numpy", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()),
                              (Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type())))
def assign(object_to, object_from):
    assign_base(object_to, object_from)
