# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from RBniCS.backends.online import OnlineMatrix, OnlineVector

def transpose(arg, backend, wrapping):
    assert isinstance(arg, (backend.Function_Type, backend.FunctionsList, backend.Vector_Type))
    if isinstance(arg, arg, backend.FunctionsList):
        return FunctionsList_Transpose(arg, backend, wrapping)
    elif isinstance(arg, (backend.Function_Type, backend.Vector_Type)):
        return Vector_Transpose(arg, backend, wrapping)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in transpose.")
        
# Auxiliary class: transpose of a vector
class Vector_Transpose(object):
    def __init__(self, vector, backend, wrapping):
        assert isinstance(vector, (backend.Function_Type, backend.Vector_Type))
        self.vector = vector
        self.backend = backend
        self.wrapping = wrapping
            
    def __mul__(self, matrix_or_vector):
        assert isinstance(matrix_or_vector, (self.backend.Matrix_Type, self.backend.Function_Type, self.backend.Vector_Type))
        if isinstance(matrix_or_vector, self.backend.Matrix_Type):
            return Vector_Transpose__times__Matrix(self.vector, matrix_or_vector, self.backend, self.wrapping)
        elif isinstance(matrix_or_vector, (self.backend.Function_Type, self.backend.Vector_Type):
            self.wrapping.vector_mul_vector(self.vector, matrix_or_vector)
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in Vector_Transpose.__mul__.")
            
# Auxiliary class: multiplication of the transpose of a Vector with a Matrix
class Vector_Transpose__times__Matrix(object):
    def __init__(self, vector, matrix, backend, wrapping):
        assert isinstance(vector, (backend.Function_Type, backend.Vector_Type))
        assert isinstance(matrix, backend.Matrix_Type)
        self.vector = vector
        self.matrix = matrix
        self.backend = backend
        self.wrapping = wrapping
        
    def __mul__(self, other_vector):
        assert isinstance(other_vector, (self.backend.Function_Type, self.backend.Vector_Type))
        return self.wrapping.vector_mul_vector(self.vector, self.wrapping.matrix_mul_vector(self.matrix, other_vector))
        
# Auxiliary class: transpose of a FunctionsList
class FunctionsList_Transpose(object):
    def __init__(self, functions_list, backend, wrapping):
        assert isinstance(functions_list, backend.FunctionsList)
        self.functions_list = functions_list
        self.backend = backend
        self.wrapping = wrapping
    
    def __mul__(self, matrix_or_vector):
        assert isinstance(matrix_or_vector, (self.backend.Matrix_Type, self.backend.Function_Type, self.backend.Vector_Type))
        if isinstance(matrix_or_vector, self.backend.Matrix_Type):
            return FunctionsList_Transpose__times__Matrix(self.functions_list, matrix_or_vector, self.backend, self.wrapping)
        elif isinstance(matrix_or_vector, (self.backend.Function_Type, self.backend.Vector_Type)):
            dim = len(self.functions_list)
            online_vector = OnlineVector(dim)
            for i in range(dim):
                online_vector[i] = self.wrapping.vector_mul_vector(self.functions_list[i], matrix_or_vector)
            return online_vector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList_Transpose.__mul__.")
            
# Auxiliary class: multiplication of the transpose of a FunctionsList with a Matrix
class FunctionsList_Transpose__times__Matrix(object):
    def __init__(self, functions_list, matrix, backend, wrapping):
        assert isinstance(functions_list, backend.FunctionsList)
        assert isinstance(matrix, backend.Matrix_Type)
        self.functions_list = functions_list
        self.matrix = matrix
        self.backend = backend
        self.wrapping = wrapping
        
    # self * functionsList2 [used e.g. to compute Z^T*A*Z or S^T*X*S (return OnlineMatrix), or Riesz_A^T*X*Riesz_F (return OnlineVector)]
    def __mul__(self, other_functions_list__or__function):
        assert isinstance(other_functions_list__or__function, (self.backend.FunctionsList, self.backend.Function_Type, self.backend.Vector_Type)
        if isinstance(other_functions_list__or__function, self.backend.FunctionsList):
            other_functions_list = other_functions_list__or__vector
            assert len(self.functions_list) == len(other_functions_list)
            dim = len(self.functions_list)
            online_matrix = OnlineMatrix(dim, dim)
            for j in range(dim):
                matrix_times_function_j = self.wrapping.matrix_mul_vector(self.matrix, self.functions_list[j])
                for i in range(dim):
                    online_matrix[i, j] = self.wrapping.vector_mul_vector(self.functions_list[i], matrix_times_function_j)
            return online_matrix
        elif isinstance(other_functions_list__or__function, (self.backend.Function_Type, self.backend.Vector_Type)):
            function = other_functions_list__or__vector
            dim = len(self.functions_list)
            online_vector = OnlineVector(dim)
            matrix_times_function = self.wrapping.matrix_mul_vector(self.matrix, function)
            for i in range(dim):
                online_vector[i] = self.wrapping.vector_mul_vector(self.functionsList[i], matrix_times_function)
            return online_vector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in FunctionsList_Transpose__times__Matrix.__mul__.")
        
