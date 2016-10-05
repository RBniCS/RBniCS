# Copyright (C) 2015-2016 by the RBniCS authors
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

from dolfin import Argument, as_backend_type, assemble, Point, project
from ufl import replace, replace_integral_domains
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.algorithms.traversal import iter_expressions
from RBniCS.backends.fenics.matrix import Matrix
from RBniCS.backends.fenics.vector import Vector
from RBniCS.backends.fenics.function import Function
from RBniCS.backends.fenics.functions_list import FunctionsList
from RBniCS.backends.fenics.tensors_list import TensorsList
from RBniCS.backends.fenics.projected_parametrized_tensor import ProjectedParametrizedTensor
from RBniCS.backends.fenics.projected_parametrized_expression import ProjectedParametrizedExpression
from RBniCS.backends.fenics.reduced_mesh import ReducedMesh
from RBniCS.backends.fenics.reduced_vertices import ReducedVertices
from RBniCS.backends.online import OnlineMatrix, OnlineVector
from RBniCS.utils.decorators import backend_for, tuple_of
from numpy import zeros as array, ndarray as PointType, prod as tuple_product
from mpi4py.MPI import FLOAT, MAX

# Evaluate a parametrized expression, possibly at a specific location
@backend_for("FEniCS", inputs=((Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ProjectedParametrizedTensor, ProjectedParametrizedExpression), (ReducedMesh, ReducedVertices, None)))
def evaluate(expression_, at=None):
    assert isinstance(expression_, (Matrix.Type(), Vector.Type(), Function.Type(), TensorsList, FunctionsList, ProjectedParametrizedTensor, ProjectedParametrizedExpression))
    assert at is None or isinstance(at, (ReducedMesh, ReducedVertices))
    if isinstance(expression_, (Function.Type(), FunctionsList, ProjectedParametrizedExpression)):
        assert at is None or isinstance(at, ReducedVertices)
        if isinstance(expression_, Function.Type()):
            function = expression_
            assert at is not None
            reduced_vertices = at
            reduced_vertices_list = reduced_vertices._vertex_list
            len_reduced_vertices = len(reduced_vertices_list)
            out_size = deduce_online_size_from_ufl(reduced_vertices, function)
            out = OnlineVector(out_size)
            mpi_comm = function.ufl_function_space().mesh().mpi_comm().tompi4py()
            for (index, vertex) in enumerate(reduced_vertices_list):
                out_index = None
                out_index_processor = -1
                if reduced_vertices.is_local(index):
                    out_index = function(vertex)
                    out_index_processor = mpi_comm.rank
                out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
                assert out_index_processor >= 0
                try:
                    len_out_index = len(out_index)
                except TypeError: # no attribute len, so it was a scalar function
                    out[index] = mpi_comm.bcast(out_index, root=out_index_processor)
                else: # it was a vector or tensor function
                    mpi_comm.Bcast([out_index, FLOAT], root=out_index_processor)
                    for (component, value) in out_index:
                        out[index + len_reduced_vertices*component] = value # block vector, each block corresponds to a component
            return out
        elif isinstance(expression_, FunctionsList):
            functions_list = expression_
            assert at is not None
            reduced_vertices = at
            len_reduced_vertices = len(reduced_vertices._vertex_list)
            out_size = deduce_online_size_from_ufl(reduced_vertices, functions_list[0])
            out = OnlineMatrix(out_size, out_size)
            for (j, fun_j) in enumerate(functions_list):
                evaluate_fun_j = evaluate(fun_j, at)
                for (i, out_ij) in enumerate(evaluate_fun_j):
                    out[i, j + (i//len_reduced_vertices)*len_reduced_vertices] = out_ij # block diagonal matrix, each block corresponds to a component
            return out
        elif isinstance(expression_, ProjectedParametrizedExpression):
            expression = expression_._expression
            if at is None:
                space = expression_._space
                return project(expression, space)
            else:
                assert at is not None
                reduced_vertices = at
                reduced_vertices_list = reduced_vertices._vertex_list
                len_reduced_vertices = len(reduced_vertices_list)
                out_size = deduce_online_size_from_ufl(reduced_vertices, expression)
                out = OnlineVector(out_size)
                mpi_comm = expression_._space.mesh().mpi_comm().tompi4py()
                for (index, vertex) in enumerate(reduced_vertices_list):
                    out_index = array(expression.value_size())
                    out_index_processor = -1
                    if reduced_vertices.is_local(index):
                        expression.eval(out_index, vertex)
                        out_index_processor = mpi_comm.rank
                    out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
                    assert out_index_processor >= 0
                    mpi_comm.Bcast([out_index, FLOAT], root=out_index_processor)
                    for (component, value) in enumerate(out_index):
                        out[index + len_reduced_vertices*component] = value # block vector, each block corresponds to a component
                return out
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to evaluate")
    elif isinstance(expression_, (Matrix.Type(), Vector.Type(), TensorsList, ProjectedParametrizedTensor)):
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
        elif isinstance(expression_, ProjectedParametrizedTensor):
            if at is None:
                form = expression_._form
                tensor = assemble(form)
                tensor.generator = expression_ # for I/O
                return tensor
            else:
                form = replace_test_trial_functions(expression_._form, at.get_reduced_function_space())
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

# HELPER FUNCTIONS
def deduce_online_size_from_ufl(reduced_vertices, expression):
    components = tuple_product(expression.ufl_shape)
    assert (
        isinstance(components, int) 
            or 
        # numpy returns the float 1.0 for empty tuple [scalar functions]
        (isinstance(components, float) and components == 1.0)
    )
    return len(reduced_vertices._vertex_list)*int(components)
    
def evaluate_and_vectorize_sparse_matrix_at_dofs(sparse_matrix, dofs_list):
    mat = as_backend_type(sparse_matrix).mat()
    row_start, row_end = mat.getOwnershipRange()
    out_size = len(dofs_list)
    out = OnlineVector(out_size)
    mpi_comm = mat.comm.tompi4py()
    for (index, dofs) in enumerate(dofs_list):
        i = dofs[0]
        out_index = None
        out_index_processor = -1
        if i >= row_start and i < row_end:
            j = dofs[1]
            out_index = mat.getValue(i, j)
            out_index_processor = mpi_comm.rank
        out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
        assert out_index_processor >= 0
        out[index] += mpi_comm.bcast(out_index, root=out_index_processor)
    return out
    
def evaluate_sparse_vector_at_dofs(sparse_vector, dofs_list):
    vec = as_backend_type(sparse_vector).vec()
    row_start, row_end = vec.getOwnershipRange()
    out_size = len(dofs_list)
    out = OnlineVector(out_size)
    mpi_comm = vec.comm.tompi4py()
    for (index, dofs) in enumerate(dofs_list):
        i = dofs[0]
        out_index = None
        out_index_processor = -1
        if i >= row_start and i < row_end:
            out_index = vec.getValue(i)
            out_index_processor = mpi_comm.rank
        out_index_processor = mpi_comm.allreduce(out_index_processor, op=MAX)
        assert out_index_processor >= 0
        out[index] += mpi_comm.bcast(out_index, root=out_index_processor)
    return out
    
def replace_test_trial_functions(form, reduced_V):
    if (form, reduced_V) not in replace_test_trial_functions__cache:
        trial_test_functions_replacements = dict()
        
        for integral in form.integrals():
            for expression in iter_expressions(integral):
                for node in traverse_unique_terminals(expression):
                    if isinstance(node, Argument) and node not in trial_test_functions_replacements:
                        trial_test_functions_replacements[node] = Argument(reduced_V, node.number(), node.part())
        
        replace_test_trial_functions__cache[(form, reduced_V)] = \
            replace_integral_domains(replace(form, trial_test_functions_replacements), reduced_V.mesh().ufl_domain())
            
    return replace_test_trial_functions__cache[(form, reduced_V)]
replace_test_trial_functions__cache = dict()
    
