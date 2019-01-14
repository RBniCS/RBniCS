# Copyright (C) 2015-2019 by the RBniCS authors
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

from ufl.core.operator import Operator
from rbnics.backends.basic import transpose as basic_transpose
from rbnics.backends.dolfin.basis_functions_matrix import BasisFunctionsMatrix
from rbnics.backends.dolfin.evaluate import evaluate
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.non_affine_expansion_storage import NonAffineExpansionStorage
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
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

backend = ModuleWrapper(BasisFunctionsMatrix, evaluate, Function, FunctionsList, Matrix, NonAffineExpansionStorage, ParametrizedTensorFactory, TensorsList, Vector)
wrapping = ModuleWrapper(function_to_vector, matrix_mul_vector, vector_mul_vector, vectorized_matrix_inner_vectorized_matrix)
online_backend = ModuleWrapper(OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper()
transpose_base = basic_transpose(backend, wrapping, online_backend, online_wrapping, AdditionalIsFunction, ConvertAdditionalFunctionTypes)

@backend_for("dolfin", inputs=((BasisFunctionsMatrix, Function.Type(), FunctionsList, Matrix.Type(), Operator, ParametrizedTensorFactory, TensorsList, Vector.Type()), ))
def transpose(arg):
    return transpose_base(arg)
