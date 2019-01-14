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

from numpy import argmax, abs as numpy_abs, unravel_index
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, overload

# abs function to compute maximum absolute value of an expression, matrix or vector (for EIM). To be used in combination with max,
# even though here we actually carry out both the max and the abs!
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), ))
def abs(expression):
    return _abs(expression)
    
@overload
def _abs(matrix: Matrix.Type()):
    abs_matrix = numpy_abs(matrix)
    (i_max, j_max) = unravel_index(argmax(abs_matrix), abs_matrix.shape)
    (i_max, j_max) = (int(i_max), int(j_max)) # numpy.intXX types are not subclasses of int, but can be converted to int
    return AbsOutput(matrix[(i_max, j_max)], (i_max, j_max))
    
@overload
def _abs(vector: Vector.Type()):
    abs_vector = numpy_abs(vector)
    i_max = argmax(abs_vector)
    i_max = int(i_max) # numpy.intXX types are not subclasses of int, but can be converted to int
    return AbsOutput(vector[i_max], (i_max, ))
    
# Auxiliary class to signal to the max() function that it is dealing with an output of the abs() method
class AbsOutput(object):
    def __init__(self, max_abs_return_value, max_abs_return_location):
        self.max_abs_return_value = max_abs_return_value
        self.max_abs_return_location = max_abs_return_location
