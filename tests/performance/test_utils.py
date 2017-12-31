# Copyright (C) 2015-2018 by the RBniCS authors
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

from dolfin import Function
from numpy.random import random, randint
from rbnics.backends.online.numpy import Matrix as NumpyMatrix, Vector as NumpyVector

def RandomDolfinFunction(V):
    f = Function(V)
    f.vector().set_local(_rand(f.vector().get_local().size))
    f.vector().apply("insert")
    return f

def RandomNumber():
    return _rand(1)[0]

def RandomNumpyMatrix(N, M):
    m = NumpyMatrix(N, M)
    m[:, :] = _rand(N, M)
    return m
    
def RandomNumpyVector(N):
    v = NumpyVector(N)
    v[:] = _rand(N)
    return v
    
def RandomTuple(Q):
    return tuple(float(v) for v in _rand(Q))
                
def _rand(*args):
    return (-1)**randint(2, size=(args))*random(args)/(1e-3 + random(args))
