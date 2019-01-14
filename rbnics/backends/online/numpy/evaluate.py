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

# from rbnics.backends.online.basic import evaluate as basic_evaluate
# from rbnics.backends.online.numpy.function import Function
from rbnics.backends.online.numpy.matrix import Matrix
# from rbnics.backends.online.numpy.parametrized_expression_factory import ParametrizedExpressionFactory
# from rbnics.backends.online.numpy.parametrized_tensor_factory import ParametrizedTensorFactory
# from rbnics.backends.online.numpy.reduced_mesh import ReducedMesh
# from rbnics.backends.online.numpy.reduced_vertices import ReducedVertices
# from rbnics.backends.online.numpy.tensors_list import TensorsList
from rbnics.backends.online.numpy.vector import Vector
from rbnics.utils.decorators import backend_for, tuple_of

# backend = ModuleWrapper(Function, FunctionsList, Matrix, ParametrizedExpressionFactory, ParametrizedTensorFactory, ReducedMesh, ReducedVertices, TensorsList, Vector)
# wrapping = ModuleWrapper(evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs, evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh, expression_on_truth_mesh, form_on_reduced_function_space, form_on_truth_function_space)
# online_backend = ModuleWrapper(OnlineFunction=Function, OnlineMatrix=Matrix, OnlineVector=Vector)
# online_wrapping = ModuleWrapper()
# evaluate_base = basic_evaluate(backend, wrapping, online_backend, online_wrapping)
evaluate_base = None # TODO

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), (tuple_of(int), tuple_of(tuple_of(int)), None)))
def evaluate(expression, at=None):
    return evaluate_base(expression, at)
