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
from ufl import Form
from ufl.core.operator import Operator
from dolfin import as_backend_type, assemble
import rbnics.backends # avoid circular imports when importing fenics backend
from rbnics.backends.fenics.wrapping import function_from_ufl_operators

def matrix_mul_vector(matrix, vector):
    if isinstance(vector, (rbnics.backends.fenics.Function.Type(), Operator)):
        vector = function_from_ufl_operators(vector).vector()
    elif isinstance(vector, Form):
        assert len(vector.arguments()) is 1
        vector = assemble(vector)
    if isinstance(matrix, Form):
        assert len(matrix.arguments()) is 2
        matrix = assemble(matrix)
    return matrix*vector

def vectorized_matrix_inner_vectorized_matrix(matrix, other_matrix):
    assert isinstance(matrix, rbnics.backends.fenics.Matrix.Type()) or (isinstance(matrix, Form) and len(matrix.arguments()) is 2)
    assert isinstance(other_matrix, rbnics.backends.fenics.Matrix.Type()) or (isinstance(other_matrix, Form) and len(other_matrix.arguments()) is 2)
    if isinstance(matrix, Form) and len(matrix.arguments()) is 2:
        matrix = assemble(matrix)
    if isinstance(other_matrix, Form) and len(other_matrix.arguments()) is 2:
        other_matrix = assemble(other_matrix)
    matrix = as_backend_type(matrix).mat()
    other_matrix = as_backend_type(other_matrix).mat()
    mat = matrix.transposeMatMult(other_matrix)
    # petsc4py does not expose MatGetTrace, we do this by hand
    return mat.getDiagonal().sum()
