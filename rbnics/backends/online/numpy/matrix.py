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

from numpy import matrix as MatrixBaseType, zeros as _MatrixContent_Base
from rbnics.backends.online.basic import Matrix as BasicMatrix
import rbnics.backends.online.numpy
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, OnlineSizeType

class _Matrix_Type_Base(MatrixBaseType):
    def __mul__(self, other):
        if isinstance(other, Vector.Type()):
            mat_times_other = MatrixBaseType.__mul__(self, other)
            output = Vector(self.M)
            output[:] = mat_times_other[:, 0]
            return output
        else:
            return MatrixBaseType.__mul__(self, other)
            
_Matrix_Type = BasicMatrix(_Matrix_Type_Base)
    
@backend_for("numpy", inputs=(OnlineSizeType, OnlineSizeType), output=_Matrix_Type)
def Matrix(M, N):
    (M_sum, N_sum) = _Matrix_Type.convert_matrix_sizes_from_dicts(M, N)
    output = _Matrix_Type(_MatrixContent_Base((M_sum, N_sum)))
    output.M = M
    output.N = N
    output.backend = rbnics.backends.online.numpy
    output.wrapping = rbnics.backends.online.numpy.wrapping
    return output
