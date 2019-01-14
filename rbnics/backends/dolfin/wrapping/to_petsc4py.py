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
