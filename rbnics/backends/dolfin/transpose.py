# Copyright (C) 2015-2018 by the RBniCS authors
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

from ufl import Form
from ufl.core.operator import Operator
from dolfin import assemble
from rbnics.backends.basic import transpose as basic_transpose
from rbnics.backends.dolfin.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.tensors_list import TensorsList
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import function_from_ufl_operators, function_to_vector, matrix_mul_vector, vector_mul_vector, vectorized_matrix_inner_vectorized_matrix
from rbnics.backends.online import OnlineMatrix, OnlineVector
from rbnics.utils.decorators import backend_for, ModuleWrapper

def AdditionalIsFunction(arg):
    return isinstance(arg, Operator)
def ConvertAdditionalFunctionTypes(arg):
    assert isinstance(arg, Operator)
    return function_from_ufl_operators(arg)
    
def IsForm(arg):
    return isinstance(arg, Form)
def IsParametrizedTensorFactory(arg):
    from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory # cannot import at global scope due to cyclic dependence
    return isinstance(arg, ParametrizedTensorFactory)
    
def AdditionalIsVector(arg):
    return (
        (IsForm(arg) and len(arg.arguments()) is 1)
            or
        (IsParametrizedTensorFactory(arg) and len(arg._form.arguments()) is 1)
    )
def ConvertAdditionalVectorTypes(arg):
    if IsForm(arg):
        assert len(arg.arguments()) is 1
        return assemble(arg)
    elif IsParametrizedTensorFactory(arg):
        assert len(arg._form.arguments()) is 1
        from rbnics.backends.dolfin.evaluate import evaluate # cannot import at global scope due to cyclic dependence
        return evaluate(arg)

def AdditionalIsMatrix(arg):
    return (
        (IsForm(arg) and len(arg.arguments()) is 2)
            or
        (IsParametrizedTensorFactory(arg) and len(arg._form.arguments()) is 2)
    )
def ConvertAdditionalMatrixTypes(arg):
    if IsForm(arg):
        assert len(arg.arguments()) is 2
        return assemble(arg)
    elif IsParametrizedTensorFactory(arg):
        assert len(arg._form.arguments()) is 2
        from rbnics.backends.dolfin.evaluate import evaluate # cannot import at global scope due to cyclic dependence
        return evaluate(arg)

backend = ModuleWrapper(BasisFunctionsMatrix, Function, FunctionsList, Matrix, TensorsList, Vector)
wrapping = ModuleWrapper(function_to_vector, matrix_mul_vector, vector_mul_vector, vectorized_matrix_inner_vectorized_matrix)
online_backend = ModuleWrapper(OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper()
transpose_base = basic_transpose(backend, wrapping, online_backend, online_wrapping, AdditionalIsFunction, ConvertAdditionalFunctionTypes, AdditionalIsVector, ConvertAdditionalVectorTypes, AdditionalIsMatrix, ConvertAdditionalMatrixTypes)

@backend_for("dolfin", inputs=((BasisFunctionsMatrix, Form, Function.Type(), FunctionsList, Matrix.Type(), Operator, TensorsList, Vector.Type()), ))
def transpose(arg):
    return transpose_base(arg)
