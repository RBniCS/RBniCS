# Copyright (C) 2015-2019 by the RBniCS authors
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
