# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from logging import DEBUG, getLogger
from rbnics.backends.basic.wrapping import DelayedTranspose
from rbnics.utils.decorators import overload

logger = getLogger("rbnics/backends/basic/transpose.py")


def transpose(backend, wrapping, online_backend, online_wrapping,
              AdditionalIsFunction=None, ConvertAdditionalFunctionTypes=None,
              AdditionalIsVector=None, ConvertAdditionalVectorTypes=None,
              AdditionalIsMatrix=None, ConvertAdditionalMatrixTypes=None):
    # Preprocess optional inputs
    if AdditionalIsFunction is None:
        def _AdditionalIsFunction(arg):
            return False
        AdditionalIsFunction = _AdditionalIsFunction
    if ConvertAdditionalFunctionTypes is None:
        def _ConvertAdditionalFunctionTypes(arg):
            raise NotImplementedError("Please implement conversion of additional function types")
        ConvertAdditionalFunctionTypes = _ConvertAdditionalFunctionTypes
    if AdditionalIsVector is None:
        def _AdditionalIsVector(arg):
            return False
        AdditionalIsVector = _AdditionalIsVector
    if ConvertAdditionalVectorTypes is None:
        def _ConvertAdditionalVectorTypes(arg):
            raise NotImplementedError("Please implement conversion of additional vector types")
        ConvertAdditionalVectorTypes = _ConvertAdditionalVectorTypes
    if AdditionalIsMatrix is None:
        def _AdditionalIsMatrix(arg):
            return False
        AdditionalIsMatrix = _AdditionalIsMatrix
    if ConvertAdditionalMatrixTypes is None:
        def _ConvertAdditionalMatrixTypes(arg):
            raise NotImplementedError("Please implement conversion of additional matrix types")
        ConvertAdditionalMatrixTypes = _ConvertAdditionalMatrixTypes

    # Prepare all possible auxiliary classes, which will be dispatched depending on the input argument
    _FunctionsList_Transpose__times__Matrix = FunctionsList_Transpose__times__Matrix(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes)
    _FunctionsList_Transpose = FunctionsList_Transpose(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes,
        AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
        _FunctionsList_Transpose__times__Matrix)
    _BasisFunctionsMatrix_Transpose__times__Matrix = BasisFunctionsMatrix_Transpose__times__Matrix(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes)
    _BasisFunctionsMatrix_Transpose = BasisFunctionsMatrix_Transpose(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes,
        AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
        _BasisFunctionsMatrix_Transpose__times__Matrix)
    _Vector_Transpose__times__Matrix = Vector_Transpose__times__Matrix(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes)
    _Vector_Transpose = Vector_Transpose(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsFunction, ConvertAdditionalFunctionTypes,
        AdditionalIsVector, ConvertAdditionalVectorTypes,
        AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
        _Vector_Transpose__times__Matrix)
    _VectorizedMatrix_Transpose = VectorizedMatrix_Transpose(
        backend, wrapping, online_backend, online_wrapping,
        AdditionalIsMatrix, ConvertAdditionalMatrixTypes)
    _TensorsList_Transpose = TensorsList_Transpose(
        backend, wrapping, online_backend, online_wrapping,
        _Vector_Transpose, _VectorizedMatrix_Transpose)

    # Start dispatching based on input argument
    class _Transpose(object):
        @overload(backend.FunctionsList, )
        def __call__(self, arg):
            return _FunctionsList_Transpose(arg)

        @overload(backend.BasisFunctionsMatrix, )
        def __call__(self, arg):
            return _BasisFunctionsMatrix_Transpose(arg)

        @overload(backend.TensorsList, )
        def __call__(self, arg):
            return _TensorsList_Transpose(arg)

        @overload((backend.Function.Type(), backend.Vector.Type()), )
        def __call__(self, arg):
            return _Vector_Transpose(arg)

        @overload(backend.Matrix.Type(), )
        def __call__(self, arg):
            return _VectorizedMatrix_Transpose(arg)

        @overload(object, )
        def __call__(self, arg):
            if AdditionalIsFunction(arg) or AdditionalIsVector(arg):
                return _Vector_Transpose(arg)
            elif AdditionalIsMatrix(arg):
                return _VectorizedMatrix_Transpose(arg)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    if hasattr(backend, "ParametrizedTensorFactory"):
        assert hasattr(backend, "evaluate")

        class _Transpose(_Transpose):
            @overload(backend.ParametrizedTensorFactory, )
            def __call__(self, arg):
                tensor = backend.evaluate(arg)
                return self.__call__(tensor)

    return _Transpose()


# Auxiliary: transpose of a vector
def Vector_Transpose(backend, wrapping, online_backend, online_wrapping,
                     AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                     AdditionalIsVector, ConvertAdditionalVectorTypes,
                     AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
                     _Vector_Transpose__times__Matrix):

    class _Vector_Transpose(object):
        @overload(backend.Function.Type(), )
        def __init__(self, function):
            self.vector = wrapping.function_to_vector(function)

        @overload(backend.Vector.Type(), )
        def __init__(self, vector):
            self.vector = vector

        @overload(object, )
        def __init__(self, vector):
            if AdditionalIsFunction(vector):
                self.vector = wrapping.function_to_vector(ConvertAdditionalFunctionTypes(vector))
            elif AdditionalIsVector(vector):
                self.vector = ConvertAdditionalVectorTypes(vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin v^T w")
            output = wrapping.vector_mul_vector(self.vector, wrapping.function_to_vector(function))
            logger.log(DEBUG, "End v^T w")
            return output

        @overload(backend.Matrix.Type(), )
        def __mul__(self, matrix):
            output = _Vector_Transpose__times__Matrix(self.vector, matrix)
            return output

        @overload(backend.Vector.Type(), )
        def __mul__(self, other_vector):
            logger.log(DEBUG, "Begin v^T w")
            output = wrapping.vector_mul_vector(self.vector, other_vector)
            logger.log(DEBUG, "End v^T w")
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsMatrix(other):
                matrix = ConvertAdditionalMatrixTypes(other)
                return self.__mul__(matrix)
            elif AdditionalIsVector(other):
                other_vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(other_vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    if hasattr(backend, "ParametrizedTensorFactory"):
        assert hasattr(backend, "evaluate")

        class _Vector_Transpose(_Vector_Transpose):
            @overload(backend.ParametrizedTensorFactory, )
            def __mul__(self, other):
                tensor = backend.evaluate(other)
                return self.__mul__(tensor)

    return _Vector_Transpose


# Auxiliary: multiplication of the transpose of a Vector with a Matrix
def Vector_Transpose__times__Matrix(backend, wrapping, online_backend, online_wrapping,
                                    AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                                    AdditionalIsVector, ConvertAdditionalVectorTypes):

    class _Vector_Transpose__times__Matrix(object):
        @overload(backend.Function.Type(), backend.Matrix.Type())
        def __init__(self, function, matrix):
            self.vector = wrapping.function_to_vector(function)
            self.matrix = matrix

        @overload(backend.Vector.Type(), backend.Matrix.Type())
        def __init__(self, vector, matrix):
            self.vector = vector
            self.matrix = matrix

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin v^T A w")
            output = wrapping.vector_mul_vector(
                self.vector, wrapping.matrix_mul_vector(self.matrix, wrapping.function_to_vector(function)))
            logger.log(DEBUG, "End v^T A w")
            return output

        @overload(backend.Vector.Type(), )
        def __mul__(self, other_vector):
            logger.log(DEBUG, "Begin v^T A w")
            output = wrapping.vector_mul_vector(self.vector, wrapping.matrix_mul_vector(self.matrix, other_vector))
            logger.log(DEBUG, "End v^T A w")
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsVector(other):
                other_vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(other_vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    return _Vector_Transpose__times__Matrix


# Auxiliary: transpose of a FunctionsList
def FunctionsList_Transpose(backend, wrapping, online_backend, online_wrapping,
                            AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                            AdditionalIsVector, ConvertAdditionalVectorTypes,
                            AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
                            _FunctionsList_Transpose__times__Matrix):

    class _FunctionsList_Transpose(object):
        @overload(backend.FunctionsList, )
        def __init__(self, functions_list):
            self.functions_list = functions_list

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin S^T w")
            output = online_backend.OnlineVector(len(self.functions_list))
            for (i, fun_i) in enumerate(self.functions_list):
                output[i] = wrapping.vector_mul_vector(
                    wrapping.function_to_vector(fun_i), wrapping.function_to_vector(function))
            logger.log(DEBUG, "End S^T w")
            return output

        @overload(backend.Matrix.Type(), )
        def __mul__(self, matrix):
            output = _FunctionsList_Transpose__times__Matrix(self.functions_list, matrix)
            return output

        @overload(backend.Vector.Type(), )
        def __mul__(self, vector):
            logger.log(DEBUG, "Begin S^T w")
            output = online_backend.OnlineVector(len(self.functions_list))
            for (i, fun_i) in enumerate(self.functions_list):
                output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), vector)
            logger.log(DEBUG, "End S^T w")
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsMatrix(other):
                matrix = ConvertAdditionalMatrixTypes(other)
                return self.__mul__(matrix)
            elif AdditionalIsVector(other):
                vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    return _FunctionsList_Transpose


# Auxiliary: multiplication of the transpose of a FunctionsList with a Matrix
def FunctionsList_Transpose__times__Matrix(backend, wrapping, online_backend, online_wrapping,
                                           AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                                           AdditionalIsVector, ConvertAdditionalVectorTypes):

    class _FunctionsList_Transpose__times__Matrix(object):
        @overload(backend.FunctionsList, backend.Matrix.Type())
        def __init__(self, functions_list, matrix):
            self.functions_list = functions_list
            self.matrix = matrix

        @overload(backend.FunctionsList)
        def __mul__(self, other_functions_list):
            logger.log(DEBUG, "Begin S^T*A*S")
            output = online_backend.OnlineMatrix(len(self.functions_list), len(other_functions_list))
            for (j, fun_j) in enumerate(other_functions_list):
                matrix_times_fun_j = wrapping.matrix_mul_vector(self.matrix, wrapping.function_to_vector(fun_j))
                for (i, fun_i) in enumerate(self.functions_list):
                    output[i, j] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), matrix_times_fun_j)
            logger.log(DEBUG, "End S^T*A*S")
            return output

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin S^T*A*v")
            output = online_backend.OnlineVector(len(self.functions_list))
            matrix_times_function = wrapping.matrix_mul_vector(self.matrix, wrapping.function_to_vector(function))
            for (i, fun_i) in enumerate(self.functions_list):
                output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), matrix_times_function)
            logger.log(DEBUG, "End S^T*A*v")
            return output

        @overload(backend.Vector.Type(), )
        def __mul__(self, vector):
            logger.log(DEBUG, "Begin S^T*A*v")
            output = online_backend.OnlineVector(len(self.functions_list))
            matrix_times_vector = wrapping.matrix_mul_vector(self.matrix, vector)
            for (i, fun_i) in enumerate(self.functions_list):
                output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), matrix_times_vector)
            logger.log(DEBUG, "End S^T*A*v")
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsVector(other):
                vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    return _FunctionsList_Transpose__times__Matrix


# Auxiliary: transpose of a BasisFunctionsMatrix
def BasisFunctionsMatrix_Transpose(backend, wrapping, online_backend, online_wrapping,
                                   AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                                   AdditionalIsVector, ConvertAdditionalVectorTypes,
                                   AdditionalIsMatrix, ConvertAdditionalMatrixTypes,
                                   _BasisFunctionsMatrix_Transpose__times__Matrix):

    class _BasisFunctionsMatrix_Transpose(object):
        @overload(backend.BasisFunctionsMatrix, )
        def __init__(self, basis_functions_matrix):
            self.basis_functions_matrix = basis_functions_matrix
            self._component_name_to_basis_component_index = (
                basis_functions_matrix._component_name_to_basis_component_index)
            self._component_name_to_basis_component_length = (
                basis_functions_matrix._component_name_to_basis_component_length)

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin Z^T w")
            output = online_backend.OnlineVector(self.basis_functions_matrix._component_name_to_basis_component_length)
            i = 0
            for component_name in self.basis_functions_matrix._components_name:
                for fun_i in self.basis_functions_matrix._components[component_name]:
                    output[i] = wrapping.vector_mul_vector(
                        wrapping.function_to_vector(fun_i), wrapping.function_to_vector(function))
                    i += 1
            logger.log(DEBUG, "End Z^T w")
            # Assert consistency of private attributes storing the order of components and their basis length.
            assert output._component_name_to_basis_component_index == self._component_name_to_basis_component_index
            assert output._component_name_to_basis_component_length == self._component_name_to_basis_component_length
            # Return
            return output

        @overload(backend.Matrix.Type(), )
        def __mul__(self, matrix):
            return _BasisFunctionsMatrix_Transpose__times__Matrix(self.basis_functions_matrix, matrix)

        @overload(backend.Vector.Type(), )
        def __mul__(self, vector):
            logger.log(DEBUG, "Begin Z^T w")
            output = online_backend.OnlineVector(self.basis_functions_matrix._component_name_to_basis_component_length)
            i = 0
            for component_name in self.basis_functions_matrix._components_name:
                for fun_i in self.basis_functions_matrix._components[component_name]:
                    output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), vector)
                    i += 1
            logger.log(DEBUG, "End Z^T w")
            # Assert consistency of private attributes storing the order of components and their basis length.
            assert output._component_name_to_basis_component_index == self._component_name_to_basis_component_index
            assert output._component_name_to_basis_component_length == self._component_name_to_basis_component_length
            # Return
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsMatrix(other):
                matrix = ConvertAdditionalMatrixTypes(other)
                return self.__mul__(matrix)
            elif AdditionalIsVector(other):
                vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    if hasattr(backend, "ParametrizedTensorFactory"):
        class _BasisFunctionsMatrix_Transpose(_BasisFunctionsMatrix_Transpose):
            @overload(backend.ParametrizedTensorFactory, )
            def __mul__(self, other):
                delayed_transpose = DelayedTranspose(self.basis_functions_matrix)
                return delayed_transpose * other

    return _BasisFunctionsMatrix_Transpose


# Auxiliary: multiplication of the transpose of a BasisFunctionsMatrix with a Matrix
def BasisFunctionsMatrix_Transpose__times__Matrix(backend, wrapping, online_backend, online_wrapping,
                                                  AdditionalIsFunction, ConvertAdditionalFunctionTypes,
                                                  AdditionalIsVector, ConvertAdditionalVectorTypes):

    class _BasisFunctionsMatrix_Transpose__times__Matrix(object):
        @overload(backend.BasisFunctionsMatrix, backend.Matrix.Type())
        def __init__(self, basis_functions_matrix, matrix):
            self.basis_functions_matrix = basis_functions_matrix
            self.matrix = matrix
            self._component_name_to_basis_component_index = (
                basis_functions_matrix._component_name_to_basis_component_index)
            self._component_name_to_basis_component_length = (
                basis_functions_matrix._component_name_to_basis_component_length)

        @overload(backend.BasisFunctionsMatrix)
        def __mul__(self, other_basis_functions_matrix):
            logger.log(DEBUG, "Begin Z^T*A*Z")
            output = online_backend.OnlineMatrix(
                self.basis_functions_matrix._component_name_to_basis_component_length,
                other_basis_functions_matrix._component_name_to_basis_component_length)
            j = 0
            for other_component_name in other_basis_functions_matrix._components_name:
                for fun_j in other_basis_functions_matrix._components[other_component_name]:
                    matrix_times_fun_j = wrapping.matrix_mul_vector(
                        self.matrix, wrapping.function_to_vector(fun_j))
                    i = 0
                    for self_component_name in self.basis_functions_matrix._components_name:
                        for fun_i in self.basis_functions_matrix._components[self_component_name]:
                            output[i, j] = wrapping.vector_mul_vector(
                                wrapping.function_to_vector(fun_i), matrix_times_fun_j)
                            i += 1
                    j += 1
            logger.log(DEBUG, "End Z^T*A*Z")
            # Assert consistency of private attributes storing the order of components and their basis length.
            assert output._component_name_to_basis_component_index == (
                self._component_name_to_basis_component_index,
                other_basis_functions_matrix._component_name_to_basis_component_index)
            assert output._component_name_to_basis_component_length == (
                self._component_name_to_basis_component_length,
                other_basis_functions_matrix._component_name_to_basis_component_length)
            # Return
            return output

        @overload(backend.Function.Type(), )
        def __mul__(self, function):
            logger.log(DEBUG, "Begin Z^T*A*v")
            output = online_backend.OnlineVector(self.basis_functions_matrix._component_name_to_basis_component_length)
            matrix_times_function = wrapping.matrix_mul_vector(self.matrix, wrapping.function_to_vector(function))
            i = 0
            for component_name in self.basis_functions_matrix._components_name:
                for fun_i in self.basis_functions_matrix._components[component_name]:
                    output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), matrix_times_function)
                    i += 1
            logger.log(DEBUG, "End Z^T*A*v")
            # Assert consistency of private attributes storing the order of components and their basis length.
            assert output._component_name_to_basis_component_index == self._component_name_to_basis_component_index
            assert output._component_name_to_basis_component_length == self._component_name_to_basis_component_length
            # Return
            return output

        @overload(backend.Vector.Type(), )
        def __mul__(self, vector):
            logger.log(DEBUG, "Begin Z^T*A*v")
            output = online_backend.OnlineVector(self.basis_functions_matrix._component_name_to_basis_component_length)
            matrix_times_vector = wrapping.matrix_mul_vector(self.matrix, vector)
            i = 0
            for component_name in self.basis_functions_matrix._components_name:
                for fun_i in self.basis_functions_matrix._components[component_name]:
                    output[i] = wrapping.vector_mul_vector(wrapping.function_to_vector(fun_i), matrix_times_vector)
                    i += 1
            logger.log(DEBUG, "End Z^T*A*v")
            # Assert consistency of private attributes storing the order of components and their basis length.
            assert output._component_name_to_basis_component_index == self._component_name_to_basis_component_index
            assert output._component_name_to_basis_component_length == self._component_name_to_basis_component_length
            # Return
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsFunction(other):
                function = ConvertAdditionalFunctionTypes(other)
                return self.__mul__(function)
            elif AdditionalIsVector(other):
                vector = ConvertAdditionalVectorTypes(other)
                return self.__mul__(vector)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    return _BasisFunctionsMatrix_Transpose__times__Matrix


# Auxiliary: transpose of a vectorized matrix (i.e. vector obtained by stacking its columns)
def VectorizedMatrix_Transpose(backend, wrapping, online_backend, online_wrapping,
                               AdditionalIsMatrix, ConvertAdditionalMatrixTypes):

    class _VectorizedMatrix_Transpose(object):
        @overload(backend.Matrix.Type(), )
        def __init__(self, matrix):
            self.matrix = matrix

        @overload(object, )
        def __init__(self, matrix):
            if AdditionalIsMatrix(matrix):
                self.matrix = ConvertAdditionalMatrixTypes(matrix)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

        @overload(backend.Matrix.Type(), )
        def __mul__(self, other_matrix):
            logger.log(DEBUG, "Begin A : B")
            output = wrapping.vectorized_matrix_inner_vectorized_matrix(self.matrix, other_matrix)
            logger.log(DEBUG, "End A : B")
            return output

        @overload(object, )
        def __mul__(self, other):
            if AdditionalIsMatrix(other):
                other_matrix = ConvertAdditionalMatrixTypes(other)
                return self.__mul__(other_matrix)
            else:
                raise RuntimeError("Invalid arguments in transpose.")

    return _VectorizedMatrix_Transpose


# Auxiliary: transpose of a TensorsList
def TensorsList_Transpose(backend, wrapping, online_backend, online_wrapping,
                          _Vector_Transpose, _VectorizedMatrix_Transpose):

    class _TensorsList_Transpose(object):
        @overload(backend.TensorsList, )
        def __init__(self, tensors_list):
            self.tensors_list = tensors_list

        @overload(backend.TensorsList, )
        def __mul__(self, other_tensors_list):
            logger.log(DEBUG, "Begin T^T S")
            assert len(self.tensors_list) == len(other_tensors_list)
            dim = len(self.tensors_list)
            output = online_backend.OnlineMatrix(dim, dim)
            for i in range(dim):
                for j in range(dim):
                    output[i, j] = self._transpose(self.tensors_list[i]) * other_tensors_list[j]
            logger.log(DEBUG, "End T^T S")
            return output

        @overload(backend.Vector.Type())
        def _transpose(self, vector):
            return _Vector_Transpose(vector)

        @overload(backend.Matrix.Type())
        def _transpose(self, matrix):
            return _VectorizedMatrix_Transpose(matrix)

    return _TensorsList_Transpose
