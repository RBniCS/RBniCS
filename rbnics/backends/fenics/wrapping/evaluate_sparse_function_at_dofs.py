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
## @file
#  @brief
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

from mpi4py.MPI import MAX
from petsc4py import PETSc
from dolfin import as_backend_type
import rbnics.backends # avoid circular imports when importing fenics backend

def evaluate_sparse_function_at_dofs(input_function, dofs_list, output_V, reduced_dofs_list):
    vec = as_backend_type(input_function.vector()).vec()
    vec_row_start, vec_row_end = vec.getOwnershipRange()
    output_function = rbnics.backends.fenics.Function(output_V)
    out = as_backend_type(output_function.vector()).vec()
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
    return output_function
    
