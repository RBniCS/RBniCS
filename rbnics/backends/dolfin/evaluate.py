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

from ufl.core.operator import Operator
from rbnics.backends.basic import evaluate as basic_evaluate
from rbnics.backends.dolfin.assign import assign
from rbnics.backends.dolfin.function import Function
from rbnics.backends.dolfin.functions_list import FunctionsList
from rbnics.backends.dolfin.matrix import Matrix
from rbnics.backends.dolfin.parametrized_expression_factory import ParametrizedExpressionFactory
from rbnics.backends.dolfin.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.dolfin.reduced_mesh import ReducedMesh
from rbnics.backends.dolfin.reduced_vertices import ReducedVertices
from rbnics.backends.dolfin.tensors_list import TensorsList
from rbnics.backends.dolfin.vector import Vector
from rbnics.backends.dolfin.wrapping import assemble, assert_lagrange_1, evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs, evaluate_sparse_vector_at_dofs, expression_iterator, expression_replace, form_argument_replace, form_iterator, form_replace, function_from_ufl_operators, get_auxiliary_problem_for_non_parametrized_function, is_problem_solution_or_problem_solution_component, is_problem_solution_or_problem_solution_component_type, solution_identify_component, solution_iterator, ufl_lagrange_interpolation
from rbnics.backends.dolfin.wrapping.expression_on_reduced_mesh import basic_expression_on_reduced_mesh
from rbnics.backends.dolfin.wrapping.expression_on_truth_mesh import basic_expression_on_truth_mesh
from rbnics.backends.dolfin.wrapping.form_on_reduced_function_space import basic_form_on_reduced_function_space
from rbnics.backends.dolfin.wrapping.form_on_truth_function_space import basic_form_on_truth_function_space
from rbnics.backends.online import OnlineFunction, OnlineMatrix, OnlineVector
from rbnics.utils.decorators import backend_for, ModuleWrapper, overload

backend = ModuleWrapper(assign, Function, FunctionsList, Matrix, ParametrizedExpressionFactory, ParametrizedTensorFactory, ReducedMesh, ReducedVertices, TensorsList, Vector)
wrapping_for_wrapping = ModuleWrapper(assemble, assert_lagrange_1, expression_iterator, expression_replace, form_argument_replace, form_iterator, form_replace, is_problem_solution_or_problem_solution_component, is_problem_solution_or_problem_solution_component_type, solution_identify_component, solution_iterator, ufl_lagrange_interpolation, get_auxiliary_problem_for_non_parametrized_function=get_auxiliary_problem_for_non_parametrized_function)
expression_on_reduced_mesh = basic_expression_on_reduced_mesh(backend, wrapping_for_wrapping)
expression_on_truth_mesh = basic_expression_on_truth_mesh(backend, wrapping_for_wrapping)
form_on_reduced_function_space = basic_form_on_reduced_function_space(backend, wrapping_for_wrapping)
form_on_truth_function_space = basic_form_on_truth_function_space(backend, wrapping_for_wrapping)
wrapping = ModuleWrapper(evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_function_at_dofs, evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh=expression_on_reduced_mesh, expression_on_truth_mesh=expression_on_truth_mesh, form_on_reduced_function_space=form_on_reduced_function_space, form_on_truth_function_space=form_on_truth_function_space)
online_backend = ModuleWrapper(OnlineFunction=OnlineFunction, OnlineMatrix=OnlineMatrix, OnlineVector=OnlineVector)
online_wrapping = ModuleWrapper()
evaluate_base = basic_evaluate(backend, wrapping, online_backend, online_wrapping)

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("dolfin", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), Operator, TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression, at=None):
    return _evaluate(expression, at)
    
@overload
def _evaluate(
    expression: (
        Matrix.Type(),
        Vector.Type(),
        Function.Type(),
        TensorsList,
        FunctionsList,
        ParametrizedTensorFactory,
        ParametrizedExpressionFactory
    ),
    at: (
        ReducedMesh,
        ReducedVertices,
        None
    ) = None
):
    return evaluate_base(expression, at)
    
@overload
def _evaluate(
    expression: Operator,
    at: (
        ReducedMesh,
        ReducedVertices,
        None
    ) = None
):
    return evaluate_base(function_from_ufl_operators(expression), at)
