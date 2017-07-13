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

import rbnics.backends.online

def evaluate(expression_, at, backend, wrapping):
    assert isinstance(expression_, (backend.Matrix.Type(), backend.Vector.Type(), backend.Function.Type(), backend.TensorsList, backend.FunctionsList, backend.ParametrizedTensorFactory, backend.ParametrizedExpressionFactory))
    assert at is None or isinstance(at, (backend.ReducedMesh, backend.ReducedVertices))
    if isinstance(expression_, (backend.Function.Type(), backend.FunctionsList, backend.ParametrizedExpressionFactory)):
        assert at is None or isinstance(at, backend.ReducedVertices)
        if isinstance(expression_, backend.Function.Type()):
            assert at is not None
            return wrapping.evaluate_sparse_function_at_dofs(expression_, at.get_dofs_list())
        elif isinstance(expression_, backend.FunctionsList):
            functions_list = expression_
            assert at is not None
            out_size = len(at.get_dofs_list())
            out = rbnics.backends.online.OnlineMatrix(out_size, out_size)
            for (j, fun_j) in enumerate(functions_list):
                evaluate_fun_j = backend.evaluate(fun_j, at)
                for (i, out_ij) in enumerate(evaluate_fun_j):
                    out[i, j] = out_ij
            return out
        elif isinstance(expression_, backend.ParametrizedExpressionFactory):
            if at is None:
                return wrapping.expression_on_truth_mesh(expression_)
            else:
                interpolated_expression = wrapping.expression_on_reduced_mesh(expression_, at)
                return wrapping.evaluate_sparse_function_at_dofs(interpolated_expression, at.get_reduced_dofs_list())
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to evaluate")
    elif isinstance(expression_, (backend.Matrix.Type(), backend.Vector.Type(), backend.TensorsList, backend.ParametrizedTensorFactory)):
        assert at is None or isinstance(at, backend.ReducedMesh)
        if isinstance(expression_, backend.Matrix.Type()):
            assert at is not None
            return wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs(expression_, at.get_dofs_list())
        elif isinstance(expression_, backend.Vector.Type()):
            assert at is not None
            return wrapping.evaluate_sparse_vector_at_dofs(expression_, at.get_dofs_list())
        elif isinstance(expression_, backend.TensorsList):
            tensors_list = expression_
            assert at is not None
            out_size = len(at.get_dofs_list())
            out = rbnics.backends.online.OnlineMatrix(out_size, out_size)
            for (j, tensor_j) in enumerate(tensors_list):
                evaluate_tensor_j = backend.evaluate(tensor_j, at)
                for (i, out_ij) in enumerate(evaluate_tensor_j):
                    out[i, j] = out_ij
            return out
        elif isinstance(expression_, backend.ParametrizedTensorFactory):
            if at is None:
                return wrapping.form_on_truth_function_space(expression_)
            else:
                (assembled_form, form_rank) = wrapping.form_on_reduced_function_space(expression_, at, backend)
                assert form_rank in (1, 2)
                if form_rank is 2:
                    return wrapping.evaluate_and_vectorize_sparse_matrix_at_dofs(assembled_form, at.get_reduced_dofs_list())
                elif form_rank is 1:
                    return wrapping.evaluate_sparse_vector_at_dofs(assembled_form, at.get_reduced_dofs_list())
                else: # impossible to arrive here anyway thanks to the assert
                    raise AssertionError("Invalid form rank")
        else: # impossible to arrive here anyway thanks to the assert
            raise AssertionError("Invalid argument to evaluate")
    else: # impossible to arrive here anyway thanks to the assert
        raise AssertionError("Invalid argument to evaluate")
        
