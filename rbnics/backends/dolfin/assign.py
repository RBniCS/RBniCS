# Copyright (C) 2015-2017 by the RBniCS authors
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

from dolfin import as_backend_type, assign as dolfin_assign
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.utils.decorators import backend_for, list_of, overload

@backend_for("dolfin", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), (Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type())))
def assign(object_to, object_from):
    _assign(object_to, object_from)
    
@overload
def _assign(object_to: Function.Type(), object_from: Function.Type()):
    if object_from is not object_to:
        dolfin_assign(object_to, object_from)
        
@overload
def _assign(object_to: list_of(Function.Type()), object_from: list_of(Function.Type())):
    if object_from is not object_to:
        del object_to[:]
        object_to.extend(object_from)
        
@overload
def _assign(object_to: Matrix.Type(), object_from: Matrix.Type()):
    if object_from is not object_to:
        as_backend_type(object_from).mat().copy(as_backend_type(object_to).mat(), as_backend_type(object_to).mat().Structure.SAME_NONZERO_PATTERN)
        
@overload
def _assign(object_to: Vector.Type(), object_from: Vector.Type()):
    if object_from is not object_to:
        as_backend_type(object_from).vec().copy(as_backend_type(object_to).vec())
