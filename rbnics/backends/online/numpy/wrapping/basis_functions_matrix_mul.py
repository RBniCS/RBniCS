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

from rbnics.backends.abstract import FunctionsList as AbstractFunctionsList

def basis_functions_matrix_mul_online_matrix(basis_functions_matrix, online_matrix, BasisFunctionsMatrixType):
    Z = basis_functions_matrix.V_or_Z
    assert isinstance(Z, AbstractFunctionsList)
    
    raise RuntimeError("TODO") # TODO
    output = FunctionsListType(Z)
    dim = online_matrix.shape[1]
    for j in range(dim):
        assert len(online_matrix[:, j]) == len(functions)
        output_j = function_copy(functions[0])
        output_j.vector()[:] = 0.
        for (i, fun_i) in enumerate(functions):
            online_matrix_ij = float(online_matrix[i, j])
            output_j.vector()[:] += fun_i.vector()*online_matrix_ij
        output.enrich(output_j)
    return output

def basis_functions_matrix_mul_online_vector(basis_functions_matrix, online_vector):
    raise RuntimeError("TODO") # TODO
    output = function_copy(functions[0])
    output.vector()[:] = 0.
    for (i, fun_i) in enumerate(functions):
        online_vector_i = float(online_vector[i])
        output.vector()[:] += fun_i.vector()*online_vector_i
    return output
