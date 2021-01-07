# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.core.operator import Operator
from dolfin import assign as dolfin_assign
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_from_ufl_operators, to_petsc4py
from rbnics.utils.decorators import backend_for, list_of, overload


@backend_for("dolfin", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()),
                               (Function.Type(), list_of(Function.Type()), Matrix.Type(), Operator, Vector.Type())))
def assign(object_to, object_from):
    _assign(object_to, object_from)


@overload
def _assign(object_to: Function.Type(), object_from: Function.Type()):
    if object_from is not object_to:
        dolfin_assign(object_to, object_from)


@overload
def _assign(object_to: Function.Type(), object_from: Operator):
    dolfin_assign(object_to, function_from_ufl_operators(object_from))


@overload
def _assign(object_to: list_of(Function.Type()), object_from: list_of(Function.Type())):
    if object_from is not object_to:
        del object_to[:]
        object_to.extend(object_from)


@overload
def _assign(object_to: Matrix.Type(), object_from: Matrix.Type()):
    if object_from is not object_to:
        to_petsc4py(object_from).copy(to_petsc4py(object_to), to_petsc4py(object_to).Structure.SAME_NONZERO_PATTERN)


@overload
def _assign(object_to: Vector.Type(), object_from: Vector.Type()):
    if object_from is not object_to:
        to_petsc4py(object_from).copy(to_petsc4py(object_to))
