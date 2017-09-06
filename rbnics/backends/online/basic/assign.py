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

def assign(object_to, object_from, backend):
    if object_from is not object_to:
        assert (
            (isinstance(object_to, backend.Function.Type()) and isinstance(object_from, backend.Function.Type()))
                or
            (isinstance(object_to, list) and isinstance(object_from, list) and isinstance(object_from[0], backend.Function.Type()))
                or
            (isinstance(object_to, backend.Matrix.Type()) and isinstance(object_from, backend.Matrix.Type()))
                or
            (isinstance(object_to, backend.Vector.Type()) and isinstance(object_from, backend.Vector.Type()))
        )
        if isinstance(object_to, backend.Function.Type()) and isinstance(object_from, backend.Function.Type()):
            assert object_to.vector().N == object_from.vector().N
            object_to.vector()[:] = object_from.vector()
        elif isinstance(object_to, list) and isinstance(object_from, list) and isinstance(object_from[0], backend.Function.Type()):
            del object_to[:]
            object_to.extend(object_from)
        elif isinstance(object_to, backend.Vector.Type()) and isinstance(object_from, backend.Vector.Type()):
            assert object_to.N == object_from.N
            object_to[:] = object_from
        elif isinstance(object_to, backend.Matrix.Type()) and isinstance(object_from, backend.Matrix.Type()):
            assert object_to.N == object_from.N
            assert object_to.M == object_from.M
            object_to[:, :] = object_from
        else:
            raise AssertionError("Invalid arguments to assign")
            
