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

from mpi4py.MPI import MAX
from petsc4py import PETSc
from dolfin import as_backend_type, Function
from rbnics.backends.dolfin.wrapping.evaluate_sparse_vector_at_dofs import evaluate_sparse_vector_at_dofs
from rbnics.backends.dolfin.wrapping.ufl_lagrange_interpolation import assert_lagrange_1

def evaluate_sparse_function_at_dofs(input_function, dofs_list, output_V=None, reduced_dofs_list=None):
    assert (
        (output_V is None)
            ==
        (reduced_dofs_list is None)
    )
    if output_V is None:
        assert_lagrange_1(input_function.function_space())
        return evaluate_sparse_vector_at_dofs(input_function.vector(), dofs_list)
    else:
        vec = as_backend_type(input_function.vector()).vec()
        output_function = Function(output_V)
        out = as_backend_type(output_function.vector()).vec()
        _evaluate_sparse_function_at_dofs(vec, dofs_list, out, reduced_dofs_list)
        return output_function
    
def _evaluate_sparse_function_at_dofs(vec, dofs_list, out, reduced_dofs_list):
    vec_row_start, vec_row_end = vec.getOwnershipRange()
    out_row_start, out_row_end = out.getOwnershipRange()
    mpi_comm = vec.comm.tompi4py()
    for (i, reduced_i) in zip(dofs_list, reduced_dofs_list):
        out_index = None
        vec_i_processor = -1
        if i >= vec_row_start and i < vec_row_end:
            out_index = vec.getValue(i)
            vec_i_processor = mpi_comm.rank
        vec_i_processor = mpi_comm.allreduce(vec_i_processor, op=MAX)
        assert vec_i_processor >= 0
        out_reduced_i_processor = -1
        if reduced_i >= out_row_start and reduced_i < out_row_end:
            out_reduced_i_processor = mpi_comm.rank
        out_reduced_i_processor = mpi_comm.allreduce(vec_i_processor, op=MAX)
        assert out_reduced_i_processor >= 0
        if mpi_comm.rank == vec_i_processor:
            mpi_comm.send(out_index, dest=out_reduced_i_processor)
        if mpi_comm.rank == out_reduced_i_processor:
            out.setValues(reduced_i, mpi_comm.recv(source=vec_i_processor), addv=PETSc.InsertMode.INSERT)
    out.assemble()
    out.ghostUpdate()
    