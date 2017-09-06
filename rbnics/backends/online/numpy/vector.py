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

from numpy import matrix as VectorBaseType, zeros as _VectorContent_Base
from rbnics.backends.online.basic import Vector as BasicVector
import rbnics.backends.online.numpy
from rbnics.utils.decorators import backend_for, OnlineSizeType

_Vector_Type = BasicVector(VectorBaseType)

@backend_for("numpy", inputs=(OnlineSizeType, ), output=_Vector_Type)
def Vector(N):
    N_sum = _Vector_Type.convert_vector_size_from_dict(N)
    output = _Vector_Type(_VectorContent_Base(N_sum)).transpose() # as column vector
    output.N = N
    output.backend = rbnics.backends.online.numpy
    output.wrapping = rbnics.backends.online.numpy.wrapping
    return output
