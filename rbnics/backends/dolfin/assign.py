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

from dolfin import assign as dolfin_assign
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.vector import Vector
from rbnics.utils.decorators import backend_for, list_of

@backend_for("dolfin", inputs=((Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type()), (Function.Type(), list_of(Function.Type()), Matrix.Type(), Vector.Type())))
def assign(object_to, object_from):
    if object_from is not object_to:
        assert (
            (isinstance(object_to, Function.Type()) and isinstance(object_from, Function.Type()))
                or
            (isinstance(object_to, list) and isinstance(object_from, list) and isinstance(object_from[0], Function.Type()))
                or
            (isinstance(object_to, Matrix.Type()) and isinstance(object_from, Matrix.Type()))
                or
            (isinstance(object_to, Vector.Type()) and isinstance(object_from, Vector.Type()))
        )
        if isinstance(object_to, Function.Type()) and isinstance(object_from, Function.Type()):
            dolfin_assign(object_to, object_from)
        elif isinstance(object_to, list) and isinstance(object_from, list) and isinstance(object_from[0], Function.Type()):
            del object_to[:]
            object_to.extend(object_from)
        elif (isinstance(object_to, Matrix.Type()) and isinstance(object_from, Matrix.Type())):
            as_backend_type(object_from).mat().copy(as_backend_type(object_to).mat(), as_backend_type(object_to).mat().Structure.SAME_NONZERO_PATTERN)
        elif (isinstance(object_to, Vector.Type()) and isinstance(object_from, Vector.Type())):
            as_backend_type(object_from).vec().copy(as_backend_type(object_to).vec())
        else:
            raise AssertionError("Invalid arguments to assign")
            
