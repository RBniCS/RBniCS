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
## @file product.py
#  @brief product function to assemble truth/reduced affine expansions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from mpi4py.MPI import MAX
from numpy import ndarray as VectorMatrixType, isscalar
from ufl.core.operator import Operator
from dolfin import Argument, assemble, Expression
from rbnics.backends.fenics.function import Function
from rbnics.backends.fenics.functions_list import FunctionsList
from rbnics.backends.fenics.matrix import Matrix
from rbnics.backends.fenics.parametrized_expression_factory import ParametrizedExpressionFactory
from rbnics.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from rbnics.backends.fenics.reduced_mesh import ReducedMesh
from rbnics.backends.fenics.reduced_vertices import ReducedVertices
from rbnics.backends.fenics.tensors_list import TensorsList
from rbnics.backends.fenics.vector import Vector
from rbnics.backends.fenics.wrapping import assert_lagrange_1, evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh, expression_on_truth_mesh, form_on_reduced_function_space, form_on_truth_function_space, ufl_lagrange_interpolation
from rbnics.backends.online import OnlineMatrix, OnlineVector
from rbnics.utils.decorators import backend_for

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("fenics", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression_, at=None):
    assert isinstance(expression_, (Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory))
    assert at is None or isinstance(at, (ReducedMesh, ReducedVertices))
    if isinstance(expression_, (Function.Type(), FunctionsList, ParametrizedExpressionFactory)):
        assert at is None or isinstance(at, ReducedVertices)
        if isinstance(expression_, Function.Type()):
            assert at is not None
            assert_lagrange_1(expression_.function_space())
            return evaluate_sparse_vector_at_dofs(expression_.vector(), at.get_dofs_list())
        elif isinstance(expression_, FunctionsList):
            functions_list = expression_
            assert at is not None
            out_size = len(at.get_dofs_list())
            out = OnlineMatrix(out_size, out_size)
            for (j, fun_j) in enumerate(functions_list):
                evaluate_fun_j = evaluate(fun_j, at)
                for (i, out_ij) in enumerate(evaluate_fun_j):
                    out[i, j] = out_ij
            return out
        elif isinstance(expression_, ParametrizedExpressionFactory):
            if at is None:
                expression = expression_on_truth_mesh(expression_)
                space = expression_._space
                assert_lagrange_1(space)
                interpolated_expression = Function(space)
                ufl_lagrange_interpolation(interpolated_expression, expression)
                return interpolated_expression
            else:
                expression = expression_on_reduced_mesh(expression_, at)
                reduced_space = at.get_reduced_function_space()
                assert_lagrange_1(reduced_space)
                interpolated_expression = Function(reduced_space)
                ufl_lagrange_interpolation(interpolated_expression, expression)
                return evaluate_sparse_vector_at_dofs(interpolated_expression.vector(), at.get_reduced_dofs_list())
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to evaluate")
    elif isinstance(expression_, (Matrix.Type(), Vector.Type(), TensorsList, ParametrizedTensorFactory)):
        assert at is None or isinstance(at, ReducedMesh)
        if isinstance(expression_, Matrix.Type()):
            assert at is not None
            return evaluate_and_vectorize_sparse_matrix_at_dofs(expression_, at.get_dofs_list())
        elif isinstance(expression_, Vector.Type()):
            assert at is not None
            return evaluate_sparse_vector_at_dofs(expression_, at.get_dofs_list())
        elif isinstance(expression_, TensorsList):
            tensors_list = expression_
            assert at is not None
            out_size = len(at.get_dofs_list())
            out = OnlineMatrix(out_size, out_size)
            for (j, tensor_j) in enumerate(tensors_list):
                evaluate_tensor_j = evaluate(tensor_j, at)
                for (i, out_ij) in enumerate(evaluate_tensor_j):
                    out[i, j] = out_ij
            return out
        elif isinstance(expression_, ParametrizedTensorFactory):
            if at is None:
                form = form_on_truth_function_space(expression_)
                tensor = assemble(form)
                tensor.generator = expression_ # for I/O
                return tensor
            else:
                form = form_on_reduced_function_space(expression_, at)
                dofs = at.get_reduced_dofs_list()
                sparse_out = assemble(form)
                form_rank = len(form.arguments())
                assert form_rank in (1, 2)
                if form_rank is 2:
                    return evaluate_and_vectorize_sparse_matrix_at_dofs(sparse_out, dofs)
                elif form_rank is 1:
                    return evaluate_sparse_vector_at_dofs(sparse_out, dofs)
                else: # impossible to arrive here anyway thanks to the assert
                    raise AssertionError("Invalid form rank")
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to evaluate")
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to evaluate")
        
