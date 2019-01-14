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

from ufl.core.operator import Operator
from dolfin import assign as dolfin_assign
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_from_ufl_operators, to_petsc4py
from rbnics.utils.decorators import backend_for, list_of, overload

@backend_for("dolfin", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), (Function.Type(), list_of(Function.Type()), Matrix.Type(), Operator, Vector.Type())))
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
def _assign(object_to: (Matrix.Type(), Vector.Type()), object_from: (Matrix.Type(), Vector.Type())):
    if object_from is not object_to:
        to_petsc4py(object_from).copy(to_petsc4py(object_to), to_petsc4py(object_to).Structure.SAME_NONZERO_PATTERN)
