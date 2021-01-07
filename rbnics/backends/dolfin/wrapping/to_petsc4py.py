# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import as_backend_type, Function
from dolfin.cpp.la import GenericMatrix, GenericVector
from rbnics.utils.decorators import overload


@overload
def to_petsc4py(function: Function):
    return to_petsc4py(function.vector())


@overload
def to_petsc4py(vector: GenericVector):
    return as_backend_type(vector).vec()


@overload
def to_petsc4py(matrix: GenericMatrix):
    return as_backend_type(matrix).mat()
