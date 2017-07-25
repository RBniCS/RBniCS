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

from petsc4py import PETSc
from dolfin import as_backend_type
import rbnics.backends.dolfin

def matrix_mul_vector(matrix, vector):
    FunctionType = rbnics.backends.dolfin.Function.Type()
    VectorType = rbnics.backends.dolfin.Vector.Type()
    assert isinstance(vector, (FunctionType, VectorType))
    if isinstance(vector, FunctionType):
        vector = vector.vector()
    assert isinstance(vector, VectorType)
    MatrixType = rbnics.backends.dolfin.Matrix.Type()
    assert isinstance(matrix, MatrixType)
    return matrix*vector

def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    MatrixType = rbnics.backends.dolfin.Matrix.Type()
    assert isinstance(matrix, MatrixType)
    assert isinstance(other_matrix, MatrixType)
    matrix = as_backend_type(matrix).mat()
    other_matrix = as_backend_type(other_matrix).mat()
    mat = matrix.transposeMatMult(other_matrix)
    # petsc4py does not expose MatGetTrace, we do this by hand
    return mat.getDiagonal().sum()
