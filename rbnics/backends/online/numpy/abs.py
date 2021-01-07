# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from numpy import argmax, abs as numpy_abs, unravel_index
from rbnics.backends.online.numpy.matrix import Matrix
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, overload


# abs function to compute maximum absolute value of an expression, matrix or vector (for EIM).
# To be used in combination with max, even though here we actually carry out both the max and the abs!
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), ))
def abs(expression):
    return _abs(expression)


@overload
def _abs(matrix: Matrix.Type()):
    abs_matrix = numpy_abs(matrix)
    (i_max, j_max) = unravel_index(argmax(abs_matrix), abs_matrix.shape)
    # i_max and j_max are of type numpy.intXX which is not a subclass of int, but can be converted to int
    (i_max, j_max) = (int(i_max), int(j_max))
    return AbsOutput(matrix[(i_max, j_max)], (i_max, j_max))


@overload
def _abs(vector: Vector.Type()):
    abs_vector = numpy_abs(vector)
    i_max = argmax(abs_vector)
    i_max = int(i_max)  # numpy.intXX types are not subclasses of int, but can be converted to int
    return AbsOutput(vector[i_max], (i_max, ))


# Auxiliary class to signal to the max() function that it is dealing with an output of the abs() method
class AbsOutput(object):
    def __init__(self, max_abs_return_value, max_abs_return_location):
        self.max_abs_return_value = max_abs_return_value
        self.max_abs_return_location = max_abs_return_location
