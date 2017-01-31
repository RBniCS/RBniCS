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
from numpy import zeros as array, ndarray as VectorMatrixType
from dolfin import Argument, assemble, project
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.parametrized_expression_factory import ParametrizedExpressionFactory
from RBniCS.backends.fenics.parametrized_tensor_factory import ParametrizedTensorFactory
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.fenics.tensors_list import TensorsList
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.wrapping import evaluate_and_vectorize_sparse_matrix_at_dofs, evaluate_sparse_vector_at_dofs, expression_on_reduced_mesh, expression_on_truth_mesh, form_on_reduced_function_space, form_on_truth_function_space
from RBniCS.backends.online import OnlineMatrix, OnlineVector
from RBniCS.utils.decorators import backend_for

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("fenics", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression_, at=None):
    assert isinstance(expression_, (Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ParametrizedTensorFactory, ParametrizedExpressionFactory))
    assert at is None or isinstance(at, (ReducedMesh, ReducedVertices))
    if isinstance(expression_, (Function.Type(), FunctionsList, ParametrizedExpressionFactory)):
        assert at is None or isinstance(at, ReducedVertices)
        if isinstance(expression_, Function.Type()):
            function = expression_
            assert at is not None
            reduced_vertices = at._vertex_list
            reduced_components = at._component_list
            assert len(reduced_vertices) == len(reduced_components)
            out_size = len(reduced_vertices)
            out = OnlineVector(out_size)
            mpi_comm = function.ufl_function_space().mesh().mpi_comm().tompi4py()
            for (index, (vertex, component)) in enumerate(zip(reduced_vertices, reduced_components)):
                out_index = None
                out_index_type = None
                out_index_processor = -1
                if at.is_local(index):
                    out_index = function(vertex)
                    out_index_processor = mpi_comm.rank
                    assert isinstance(out_index, (float, VectorMatrixType))
                    if isinstance(out_index, float):
                        out_index_type = "scalar"
                    elif isinstance(out_index, VectorMatrixType):
                        out_index_type = "vector_matrix"
                    else: # impossible to arrive here anyway thanks to the assert
                        raise AssertionError("Invalid argument to evaluate")
                out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
                assert out_index_processor >= 0
                out_index_type = mpi_comm.bcast(out_index_type, root=out_index_processor)
                assert out_index_type in ("scalar", "vector_matrix")
                if out_index_type == "scalar":
                    out[index] = mpi_comm.bcast(out_index, root=out_index_processor)
                elif out_index_type == "vector_matrix":
                    if out_index is not None: # on out_index_processor
                        out[index] = mpi_comm.bcast(out_index[component], root=out_index_processor)
                    else: # on other processors
                        out[index] = mpi_comm.bcast(None, root=out_index_processor)
                else: # impossible to arrive here anyway thanks to the assert
                    raise AssertionError("Invalid argument to evaluate")
            return out
        elif isinstance(expression_, FunctionsList):
            functions_list = expression_
            assert at is not None
            assert len(at._vertex_list) == len(at._component_list)
            out_size = len(at._vertex_list)
            out = OnlineMatrix(out_size, out_size)
            for (j, fun_j) in enumerate(functions_list):
                evaluate_fun_j = evaluate(fun_j, at)
                for (i, out_ij) in enumerate(evaluate_fun_j):
                    out[i, j] = out_ij
            return out
        elif isinstance(expression_, ParametrizedExpressionFactory):
            if at is None:
                expression = expression_on_truth_mesh(expression_._expression)
                space = expression_._space
                return project(expression, space)
            else:
                expression = expression_on_reduced_mesh(expression_._expression, at)
                reduced_vertices = at._vertex_list
                reduced_components = at._component_list
                assert len(reduced_vertices) == len(reduced_components)
                out_size = len(reduced_vertices)
                out = OnlineVector(out_size)
                mpi_comm = expression_._space.mesh().mpi_comm().tompi4py()
                for (index, (vertex, component)) in enumerate(zip(reduced_vertices, reduced_components)):
                    out_index = array(expression.value_size())
                    out_index_processor = -1
                    if at.is_local(index):
                        expression.eval(out_index, vertex)
                        out_index_processor = mpi_comm.rank
                    out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
                    assert out_index_processor >= 0
                    out[index] = mpi_comm.bcast(out_index[component], root=out_index_processor)
                return out
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
                form = form_on_truth_function_space(expression_._form)
                tensor = assemble(form)
                tensor.generator = expression_ # for I/O
                return tensor
            else:
                form = form_on_reduced_function_space(expression_._form, at)
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
        
