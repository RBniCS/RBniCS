# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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

# backend = ModuleWrapper(Function, FunctionsList, Matrix, ParametrizedExpressionFactory, ParametrizedTensorFactory,
#                         ReducedMesh, ReducedVertices, TensorsList, Vector)
# wrapping = ModuleWrapper(evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs,
#                          evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh, expression_on_truth_mesh,
#                          form_on_reduced_function_space, form_on_truth_function_space)
# online_backend = ModuleWrapper(OnlineFunction=Function, OnlineMatrix=Matrix, OnlineVector=Vector)
# online_wrapping = ModuleWrapper()
# evaluate_base = basic_evaluate(backend, wrapping, online_backend, online_wrapping)
evaluate_base = None  # TODO


# Evaluate a parametrized expression, possibly at a specific location
@backend_for("numpy", inputs=((Matrix.Type(), Vector.Type()), (tuple_of(int), tuple_of(tuple_of(int)), None)))
def evaluate(expression, at=None):
    return evaluate_base(expression, at)
