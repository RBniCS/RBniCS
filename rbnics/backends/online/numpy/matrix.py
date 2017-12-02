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

from numpy import matrix, zeros
from rbnics.backends.online.basic import Matrix as BasicMatrix
from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.vector import Vector
from rbnics.backends.online.numpy.wrapping import Slicer
from rbnics.utils.decorators import backend_for, ModuleWrapper, OnlineSizeType

def MatrixBaseType(M, N):
    return matrix(zeros((M, N)))
            
backend = ModuleWrapper(Function, Vector)
wrapping = ModuleWrapper(Slicer=Slicer)
_Matrix_Type = BasicMatrix(backend, wrapping, MatrixBaseType)
    
@backend_for("numpy", inputs=(OnlineSizeType, OnlineSizeType))
def Matrix(M, N):
    return _Matrix_Type(M, N)
    
# Attach a Type() function
def Type():
    return _Matrix_Type
Matrix.Type = Type
