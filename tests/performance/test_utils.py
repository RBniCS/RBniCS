# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# RBniCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# RBniCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with RBniCS. If not, see <http://www.gnu.org/licenses/>.
#

from mpi4py.MPI import COMM_WORLD
from dolfin import Function
from numpy.random import random, randint
from rbnics.backends.online.numpy import Matrix as NumpyMatrix, Vector as NumpyVector

def RandomDolfinFunction(V):
    f = Function(V)
    f.vector().set_local(_rand(f.vector().get_local().size))
    f.vector().apply("insert")
    return f

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
    return (-1)**randint(2, size=(args))*random(args)/(1e-3 + random(args))
