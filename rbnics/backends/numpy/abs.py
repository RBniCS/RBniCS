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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.numpy.matrix import Matrix
from RBniCS.backends.numpy.vector import Vector
from RBniCS.utils.decorators import backend_for
from numpy import argmax, abs as numpy_abs, unravel_index

# abs function to compute maximum absolute value of an expression, matrix or vector (for EIM). To be used in combination with max,
# even though here we actually carry out both the max and the abs!
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), ))
def abs(expression):
    assert isinstance(expression, (Matrix.Type(), Vector.Type()))
    if isinstance(expression, Matrix.Type()):
        matrix = expression
        abs_matrix = numpy_abs(matrix)
        ij_max = unravel_index(argmax(abs_matrix), abs_matrix.shape)
        return AbsOutput(matrix[ij_max], ij_max)
    elif isinstance(expression, Vector.Type()):
        vector = expression
        abs_vector = numpy_abs(vector)
        i_max = (argmax(abs_vector), )
        return AbsOutput(vector[i_max], i_max)
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to abs")
    
# Auxiliary class to signal to the max() function that it is dealing with an output of the abs() method
class AbsOutput(object):
    def __init__(self, max_abs_return_value, max_abs_return_location):
        self.max_abs_return_value = max_abs_return_value
        self.max_abs_return_location = max_abs_return_location
        
