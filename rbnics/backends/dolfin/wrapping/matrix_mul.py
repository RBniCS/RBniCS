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

from rbnics.backends.dolfin.wrapping.to_petsc4py import to_petsc4py

def matrix_mul_vector(matrix, vector):
    return matrix*vector

def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    matrix = to_petsc4py(matrix)
    other_matrix = to_petsc4py(other_matrix)
    # petsc4py does not expose MatGetTrace, we do this by hand
    return matrix.transposeMatMult(other_matrix).getDiagonal().sum()
