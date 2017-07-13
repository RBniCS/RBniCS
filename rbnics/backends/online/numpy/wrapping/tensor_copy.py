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

import rbnics.backends.online.numpy

def tensor_copy(tensor):
    assert isinstance(tensor, (rbnics.backends.online.numpy.Matrix.Type(), rbnics.backends.online.numpy.Vector.Type())
    if isinstance(tensor, rbnics.backends.online.numpy.Matrix.Type()):
        m = rbnics.backends.online.numpy.Matrix(*tensor.shape)
        m[:] = tensor
        return m
    elif isinstance(tensor, rbnics.backends.online.numpy.Vector.Type()):
        V = rbnics.backends.online.numpy.Vector(tensor.size)
        v[:] = tensor
        return v
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in tensor_copy.")
