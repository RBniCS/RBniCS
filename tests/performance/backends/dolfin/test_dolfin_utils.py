# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from mpi4py.MPI import COMM_WORLD
from dolfin import Function
from numpy.random import random, randint
from rbnics.backends.online.numpy import Vector as NumpyVector


def RandomDolfinFunction(V):
    f = Function(V)
    f.vector().set_local(_rand(f.vector().get_local().size))
    f.vector().apply("insert")
    return f


def RandomNumpyVector(N):
    v = NumpyVector(N)
    if COMM_WORLD.rank == 0:
        v[:] = _rand(N)
    COMM_WORLD.Bcast(v.content, root=0)
    return v


def RandomTuple(Q):
    if COMM_WORLD.rank == 0:
        tuple_ = tuple(float(v) for v in _rand(Q))
    else:
        tuple_ = None
    tuple_ = COMM_WORLD.bcast(tuple_, root=0)
    assert tuple_ is not None
    return tuple_


def _rand(*args):
    return (-1)**randint(2, size=(args)) * random(args) / (1e-3 + random(args))
