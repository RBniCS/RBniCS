# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import Function, FunctionSpace
from dolfin.cpp.la import GenericMatrix, GenericVector
from rbnics.utils.decorators import overload, tuple_of


@overload
def get_mpi_comm(function: Function):
    return get_mpi_comm(function.function_space())


@overload
def get_mpi_comm(V: FunctionSpace):
    mpi_comm = V.mesh().mpi_comm()
    return mpi_comm


@overload
def get_mpi_comm(V: tuple_of(FunctionSpace)):
    assert len(V) in (1, 2)
    return get_mpi_comm(V[0])


@overload
def get_mpi_comm(tensor: (GenericMatrix, GenericVector)):
    mpi_comm = tensor.mpi_comm()
    return mpi_comm
