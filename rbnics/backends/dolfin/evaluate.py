# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from ufl.core.operator import Operator
from rbnics.backends.basic import evaluate as basic_evaluate
from rbnics.backends.dolfin.assign import assign
from rbnics.backends.dolfin.copy import copy
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.parametrized_expression_factory import ParametrizedExpressionFactory
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.reduced_mesh import ReducedMesh
from rbnics.backends.dolfin.reduced_vertices import ReducedVertices
from rbnics.backends.dolfin.tensors_list import TensorsList
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import (assemble, evaluate_and_vectorize_sparse_matrix_at_dofs,
                                             evaluate_expression, evaluate_sparse_function_at_dofs,
                                             evaluate_sparse_vector_at_dofs, expression_iterator,
                                             expression_replace, form_argument_replace, form_iterator,
                                             form_replace, function_from_ufl_operators,
                                             get_auxiliary_problem_for_non_parametrized_function,
                                             is_problem_solution, is_problem_solution_dot,
                                             is_problem_solution_type, solution_dot_identify_component,
                                             solution_identify_component, solution_iterator)
from rbnics.backends.dolfin.wrapping.expression_on_reduced_mesh import basic_expression_on_reduced_mesh
from rbnics.backends.dolfin.wrapping.expression_on_truth_mesh import basic_expression_on_truth_mesh
from rbnics.backends.dolfin.wrapping.form_on_reduced_function_space import basic_form_on_reduced_function_space
from rbnics.backends.dolfin.wrapping.form_on_truth_function_space import basic_form_on_truth_function_space
from rbnics.backends.online import online_assign, OnlineFunction, OnlineMatrix, OnlineVector
from rbnics.utils.decorators import backend_for, ModuleWrapper, overload

backend = ModuleWrapper(
    assign, copy, Function, FunctionsList, Matrix, ParametrizedExpressionFactory, ParametrizedTensorFactory,
    ReducedMesh, ReducedVertices, TensorsList, Vector)
wrapping_for_wrapping = ModuleWrapper(
    assemble, evaluate_expression, expression_iterator, expression_replace, form_argument_replace, form_iterator,
    form_replace, is_problem_solution, is_problem_solution_dot, is_problem_solution_type,
    solution_dot_identify_component, solution_identify_component, solution_iterator,
    get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
online_backend_for_wrapping = ModuleWrapper(online_assign=online_assign, OnlineFunction=OnlineFunction)
online_wrapping_for_wrapping = ModuleWrapper()
expression_on_reduced_mesh = basic_expression_on_reduced_mesh(
    backend, wrapping_for_wrapping, online_backend_for_wrapping, online_wrapping_for_wrapping)
expression_on_truth_mesh = basic_expression_on_truth_mesh(backend, wrapping_for_wrapping)
form_on_reduced_function_space = basic_form_on_reduced_function_space(
    backend, wrapping_for_wrapping, online_backend_for_wrapping, online_wrapping_for_wrapping)
form_on_truth_function_space = basic_form_on_truth_function_space(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(
    evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs,
    evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh=expression_on_reduced_mesh,
    expression_on_truth_mesh=expression_on_truth_mesh,
    form_on_reduced_function_space=form_on_reduced_function_space,
    form_on_truth_function_space=form_on_truth_function_space)
online_backend = ModuleWrapper(OnlineFunction=OnlineFunction, OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper()
evaluate_base = basic_evaluate(backend, wrapping, online_backend, online_wrapping)


# Evaluate a parametrized expression, possibly at a specific location
@backend_for("dolfin", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), Operator, TensorsList,
                                FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory),
                               (ReducedMesh, ReducedVertices, None)))
def evaluate(expression, at=None, **kwargs):
    return _evaluate(expression, at, **kwargs)


@overload
def _evaluate(
    expression: (
        Matrix.Type(),
        Vector.Type(),
        TensorsList
    ),
    at: ReducedMesh,
    **kwargs
):
    assert len(kwargs) == 0
    return evaluate_base(expression, at)


@overload
def _evaluate(
    expression: (
        Function.Type(),
        FunctionsList
    ),
    at: (
        ReducedMesh,
        ReducedVertices
    ),
    **kwargs
):
    assert len(kwargs) == 0
    return evaluate_base(expression, at)


@overload
def _evaluate(
    expression: ParametrizedTensorFactory,
    at: None,
    **kwargs
):
    assert (len(kwargs) == 0
            or (len(kwargs) == 1 and "tensor" in kwargs))
    tensor = kwargs.get("tensor", None)
    return evaluate_base(expression, at, tensor)


@overload
def _evaluate(
    expression: ParametrizedTensorFactory,
    at: ReducedMesh,
    **kwargs
):
    assert len(kwargs) == 0
    return evaluate_base(expression, at)


@overload
def _evaluate(
    expression: ParametrizedExpressionFactory,
    at: None,
    **kwargs
):
    assert (len(kwargs) == 0
            or (len(kwargs) == 1 and "function" in kwargs))
    function = kwargs.get("function", None)
    return evaluate_base(expression, at, function)


@overload
def _evaluate(
    expression: ParametrizedExpressionFactory,
    at: ReducedVertices,
    **kwargs
):
    assert len(kwargs) == 0
    return evaluate_base(expression, at)


@overload
def _evaluate(
    expression: Operator,
    at: (
        ReducedMesh,
        ReducedVertices,
    ),
    **kwargs
):
    assert len(kwargs) == 0
    return evaluate_base(function_from_ufl_operators(expression), at)
