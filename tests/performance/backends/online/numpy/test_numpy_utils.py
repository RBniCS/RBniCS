# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from mpi4py.MPI import COMM_WORLD
from numpy.random import random, randint
from rbnics.backends.online.numpy import Matrix as NumpyMatrix, Vector as NumpyVector


def RandomNumber():
    if COMM_WORLD.rank == 0:
        number = _rand(1)[0]
    else:
        number = None
    number = COMM_WORLD.bcast(number, root=0)
    assert number is not None
    return number


def RandomNumpyMatrix(N, M):
    m = NumpyMatrix(N, M)
    if COMM_WORLD.rank == 0:
        m[:, :] = _rand(N, M)
    COMM_WORLD.Bcast(m.content, root=0)
    return m


def RandomNumpyVector(N):
    v = NumpyVector(N)
    if COMM_WORLD.rank == 0:
        v[:] = _rand(N)
    COMM_WORLD.Bcast(v.content, root=0)
    return v


def RandomSize(N_lower, N_upper):
    if COMM_WORLD.rank == 0:
        size = randint(N_lower, N_upper)
    else:
        size = None
    size = COMM_WORLD.bcast(size, root=0)
    assert size is not None
    return size


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
