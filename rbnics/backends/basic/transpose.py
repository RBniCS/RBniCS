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
## @file transpose.py
#  @brief transpose method to be used in RBniCS.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from rbnics.utils.mpi import log, PROGRESS
from rbnics.backends.basic.wrapping import functions_list_basis_functions_matrix_adapter

def transpose(arg, backend, wrapping, online_backend, AdditionalFunctionTypes=None, AdditionalIsVector=None, AdditionalIsMatrix=None):
    if AdditionalFunctionTypes is None:
        FunctionTypes = (backend.Function.Type(), )
    else:
        FunctionTypes = AdditionalFunctionTypes + (backend.Function.Type(), )
    if AdditionalIsVector is None:
        def _AdditionalIsVector(arg):
            return False
        AdditionalIsVector = _AdditionalIsVector
    if AdditionalIsMatrix is None:
        def _AdditionalIsMatrix(arg):
            return False
        AdditionalIsMatrix = _AdditionalIsMatrix
    assert isinstance(arg, FunctionTypes + (backend.FunctionsList, backend.BasisFunctionsMatrix, backend.Vector.Type(), backend.Matrix.Type(), backend.TensorsList)) or AdditionalIsVector(arg) or AdditionalIsMatrix(arg)
    if isinstance(arg, backend.FunctionsList):
        return FunctionsList_Transpose(arg, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix)
    elif isinstance(arg, backend.BasisFunctionsMatrix):
        return BasisFunctionsMatrix_Transpose(arg, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix)
    elif isinstance(arg, backend.TensorsList):
        return TensorsList_Transpose(arg, backend, wrapping, online_backend)
    elif isinstance(arg, FunctionTypes + (backend.Vector.Type(), )) or AdditionalIsVector(arg):
        return Vector_Transpose(arg, backend, wrapping, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix)
    elif isinstance(arg, backend.Matrix.Type()) or AdditionalIsMatrix(arg):
        return VectorizedMatrix_Transpose(arg, backend, wrapping, AdditionalIsMatrix)
    else: # impossible to arrive here anyway, thanks to the assert
        raise AssertionError("Invalid arguments in transpose.")
        
# Auxiliary class: transpose of a vector
class Vector_Transpose(object):
    def __init__(self, vector, backend, wrapping, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix):
        assert isinstance(vector, FunctionTypes + (backend.Vector.Type(), )) or AdditionalIsVector(vector)
        self.vector = vector
        self.backend = backend
        self.wrapping = wrapping
        self.FunctionTypes = FunctionTypes
        self.AdditionalIsVector = AdditionalIsVector
        self.AdditionalIsMatrix = AdditionalIsMatrix
        
    def __mul__(self, matrix_or_vector):
        assert isinstance(matrix_or_vector, self.FunctionTypes + (self.backend.Matrix.Type(), self.backend.Vector.Type())) or self.AdditionalIsVector(matrix_or_vector) or self.AdditionalIsMatrix(matrix_or_vector)
        if isinstance(matrix_or_vector, self.backend.Matrix.Type()) or self.AdditionalIsMatrix(matrix_or_vector):
            return Vector_Transpose__times__Matrix(self.vector, matrix_or_vector, self.backend, self.wrapping, self.FunctionTypes, self.AdditionalIsVector, self.AdditionalIsMatrix)
        elif isinstance(matrix_or_vector, self.FunctionTypes + (self.backend.Vector.Type(), )) or self.AdditionalIsVector(matrix_or_vector):
            log(PROGRESS, "Begin v^T w")
            output = self.wrapping.vector_mul_vector(self.vector, matrix_or_vector)
            log(PROGRESS, "End v^T w")
            return output
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in Vector_Transpose.__mul__.")
            
# Auxiliary class: multiplication of the transpose of a Vector with a Matrix
class Vector_Transpose__times__Matrix(object):
    def __init__(self, vector, matrix, backend, wrapping, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix):
        assert isinstance(vector, FunctionTypes + (backend.Vector.Type(), )) or AdditionalIsVector(vector)
        assert isinstance(matrix, backend.Matrix.Type()) or AdditionalIsMatrix(matrix)
        self.vector = vector
        self.matrix = matrix
        self.backend = backend
        self.wrapping = wrapping
        self.FunctionTypes = FunctionTypes
        self.AdditionalIsVector = AdditionalIsVector
        self.AdditionalIsMatrix = AdditionalIsMatrix
        
    def __mul__(self, other_vector):
        assert isinstance(other_vector, self.FunctionTypes + (self.backend.Vector.Type(), )) or self.AdditionalIsVector(other_vector)
        log(PROGRESS, "Begin v^T A w")
        output = self.wrapping.vector_mul_vector(self.vector, self.wrapping.matrix_mul_vector(self.matrix, other_vector))
        log(PROGRESS, "End v^T A w")
        return output
        
# Auxiliary class: transpose of a FunctionsList or a BasisFunctionsMatrix
class _FunctionsList_BasisFunctionsMatrix_Transpose(object):
    def __init__(self, functions, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix, Functions_Transpose__times__Matrix):
        (self.functions_list, self.functions_list_dim) = functions_list_basis_functions_matrix_adapter(functions, backend)
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
        self.FunctionTypes = FunctionTypes
        self.AdditionalIsVector = AdditionalIsVector
        self.AdditionalIsMatrix = AdditionalIsMatrix
        self.Functions_Transpose__times__Matrix = Functions_Transpose__times__Matrix
    
    def __mul__(self, matrix_or_vector):
        assert isinstance(matrix_or_vector, self.FunctionTypes + (self.backend.Matrix.Type(), self.backend.Vector.Type())) or self.AdditionalIsVector(matrix_or_vector) or self.AdditionalIsMatrix(matrix_or_vector)
        if isinstance(matrix_or_vector, self.backend.Matrix.Type()) or self.AdditionalIsMatrix(matrix_or_vector):
            return self.Functions_Transpose__times__Matrix(self.functions_list, self.functions_list_dim, matrix_or_vector, self.backend, self.wrapping, self.online_backend, self.FunctionTypes, self.AdditionalIsVector, self.AdditionalIsMatrix)
        elif isinstance(matrix_or_vector, self.FunctionTypes + (self.backend.Vector.Type(), )) or self.AdditionalIsVector(matrix_or_vector):
            log(PROGRESS, "Begin Z^T w")
            online_vector = self.online_backend.Vector(self.functions_list_dim)
            for (i, fun_i) in enumerate(self.functions_list):
                online_vector[i] = self.wrapping.vector_mul_vector(fun_i, matrix_or_vector)
            log(PROGRESS, "End Z^T w")
            return online_vector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in _FunctionsList_BasisFunctionsMatrix_Transpose.__mul__.")
            
class FunctionsList_Transpose(_FunctionsList_BasisFunctionsMatrix_Transpose):
    def __init__(self, basis_functions_matrix, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix):
        _FunctionsList_BasisFunctionsMatrix_Transpose.__init__(self, basis_functions_matrix, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix, FunctionsList_Transpose__times__Matrix)
    
class BasisFunctionsMatrix_Transpose(_FunctionsList_BasisFunctionsMatrix_Transpose):
    def __init__(self, basis_functions_matrix, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix):
        _FunctionsList_BasisFunctionsMatrix_Transpose.__init__(self, basis_functions_matrix, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix, BasisFunctionsMatrix_Transpose__times__Matrix)
        self._basis_component_index_to_component_name = basis_functions_matrix._basis_component_index_to_component_name
        self._component_name_to_basis_component_index = basis_functions_matrix._component_name_to_basis_component_index
        self._component_name_to_basis_component_length = basis_functions_matrix._component_name_to_basis_component_length
        
    def __mul__(self, matrix_or_vector):
        output = _FunctionsList_BasisFunctionsMatrix_Transpose.__mul__(self, matrix_or_vector)
        # Attach a private attribute to output which stores the order of components and their basis length.
        # This is needed by OnlineAffineExpansionStorage when slicing. In order to preserve
        # memory for large affine expansions, in OnlineAffineExpansionStorage we will copy it
        # for the first term and then delete this attached attribute
        output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
        output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
        output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
        return output
            
# Auxiliary class: multiplication of the transpose of a FunctionsList with a Matrix
class _FunctionsList_BasisFunctionsMatrix_Transpose__times__Matrix(object):
    def __init__(self, functions_list, functions_list_dim, matrix, backend, wrapping, online_backend, FunctionTypes, AdditionalIsVector, AdditionalIsMatrix):
        self.functions_list = functions_list
        self.functions_list_dim = functions_list_dim
        assert isinstance(matrix, backend.Matrix.Type()) or AdditionalIsMatrix(matrix)
        self.matrix = matrix
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
        self.FunctionTypes = FunctionTypes
        self.AdditionalIsVector = AdditionalIsVector
        self.AdditionalIsMatrix = AdditionalIsMatrix
        
    # self * other [used e.g. to compute Z^T*A*Z or S^T*X*S (return OnlineMatrix), or Riesz_A^T*X*Riesz_F (return OnlineVector)]
    def __mul__(self, other_functions_list__or__function):
        assert isinstance(other_functions_list__or__function, self.FunctionTypes + (self.backend.BasisFunctionsMatrix, self.backend.FunctionsList, self.backend.Vector.Type())) or self.AdditionalIsVector(other_functions_list__or__function)
        if isinstance(other_functions_list__or__function, (self.backend.BasisFunctionsMatrix, self.backend.FunctionsList)):
            log(PROGRESS, "Begin Z^T*A*Z")
            (other_functions_list, other_functions_list_dim) = functions_list_basis_functions_matrix_adapter(other_functions_list__or__function, self.backend)
            online_matrix = self.online_backend.Matrix(self.functions_list_dim, other_functions_list_dim)
            for (j, fun_j) in enumerate(other_functions_list):
                matrix_times_fun_j = self.wrapping.matrix_mul_vector(self.matrix, fun_j)
                for (i, fun_i) in enumerate(self.functions_list):
                    online_matrix[i, j] = self.wrapping.vector_mul_vector(fun_i, matrix_times_fun_j)
            log(PROGRESS, "End Z^T*A*Z")
            return online_matrix
        elif isinstance(other_functions_list__or__function, self.FunctionTypes + (self.backend.Vector.Type(), )) or self.AdditionalIsVector(other_functions_list__or__function):
            log(PROGRESS, "Begin Z^T*A*v")
            function = other_functions_list__or__function
            online_vector = self.online_backend.Vector(self.functions_list_dim)
            matrix_times_function = self.wrapping.matrix_mul_vector(self.matrix, function)
            for (i, fun_i) in enumerate(self.functions_list):
                online_vector[i] = self.wrapping.vector_mul_vector(fun_i, matrix_times_function)
            log(PROGRESS, "End Z^T*A*v")
            return online_vector
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in _FunctionsList_BasisFunctionsMatrix_Transpose__times__Matrix.__mul__.")
            
class FunctionsList_Transpose__times__Matrix(_FunctionsList_BasisFunctionsMatrix_Transpose__times__Matrix):
    pass
    
class BasisFunctionsMatrix_Transpose__times__Matrix(_FunctionsList_BasisFunctionsMatrix_Transpose__times__Matrix):
    def __mul__(self, other_functions_list__or__function):
        assert isinstance(other_functions_list__or__function, self.FunctionTypes + (self.backend.BasisFunctionsMatrix, self.backend.Vector.Type())) or self.AdditionalIsVector(other_functions_list__or__function)
        output = _FunctionsList_BasisFunctionsMatrix_Transpose__times__Matrix.__mul__(self, other_functions_list__or__function)
        # Attach a private attribute to output which stores the order of components and their basis length,
        # as discussed for BasisFunctionsMatrix_Transpose.__mul__
        if isinstance(other_functions_list__or__function, self.backend.BasisFunctionsMatrix):
            output._basis_component_index_to_component_name = (self._basis_component_index_to_component_name, other_functions_list__or__function._basis_component_index_to_component_name)
            output._component_name_to_basis_component_index = (self._component_name_to_basis_component_index, other_functions_list__or__function._component_name_to_basis_component_index)
            output._component_name_to_basis_component_length = (self._component_name_to_basis_component_length, other_functions_list__or__function._component_name_to_basis_component_length)
        elif isinstance(other_functions_list__or__function, self.FunctionTypes + (self.backend.Vector.Type(), )) or self.AdditionalIsVector(other_functions_list__or__function):
            output._basis_component_index_to_component_name = self._basis_component_index_to_component_name
            output._component_name_to_basis_component_index = self._component_name_to_basis_component_index
            output._component_name_to_basis_component_length = self._component_name_to_basis_component_length
        else: # impossible to arrive here anyway, thanks to the assert
            raise AssertionError("Invalid arguments in BasisFunctionsMatrix_Transpose__times__Matrix.__mul__.")
        return output
            
# Auxiliary class: transpose of a vectorized matrix (i.e. vector obtained by stacking its columns)
class VectorizedMatrix_Transpose(object):
    def __init__(self, matrix, backend, wrapping, AdditionalIsMatrix):
        assert isinstance(matrix, backend.Matrix.Type())
        self.matrix = matrix
        self.backend = backend
        self.wrapping = wrapping
        self.AdditionalIsMatrix = AdditionalIsMatrix
            
    def __mul__(self, other_matrix):
        assert isinstance(other_matrix, self.backend.Matrix.Type()) or self.AdditionalIsMatrix(other_matrix)
        log(PROGRESS, "Begin A : B")
        output = self.wrapping.vectorized_matrix_inner_vectorized_matrix(self.matrix, other_matrix)
        log(PROGRESS, "End A : B")
        return output
        
# Auxiliary class: transpose of a TensorsList
class TensorsList_Transpose(object):
    def __init__(self, tensors_list, backend, wrapping, online_backend):
        assert isinstance(tensors_list, backend.TensorsList)
        self.tensors_list = tensors_list
        self.backend = backend
        self.wrapping = wrapping
        self.online_backend = online_backend
    
    def __mul__(self, other_tensors_list):
        log(PROGRESS, "Begin T^T S")
        assert isinstance(other_tensors_list, self.backend.TensorsList)
        assert len(self.tensors_list) == len(other_tensors_list)
        dim = len(self.tensors_list)
        online_matrix = self.online_backend.Matrix(dim, dim)
        for i in range(dim):
            for j in range(dim):
                online_matrix[i, j] = transpose(self.tensors_list[i], self.backend, self.wrapping, self.online_backend)*other_tensors_list[j]
        log(PROGRESS, "End T^T S")
        return online_matrix
        
