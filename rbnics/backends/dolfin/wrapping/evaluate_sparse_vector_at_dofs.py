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

from mpi4py.MPI import MAX
from rbnics.backends.dolfin.wrapping.to_petsc4py import to_petsc4py
from rbnics.backends.online import OnlineVector

def evaluate_sparse_vector_at_dofs(sparse_vector, dofs_list):
    vec = to_petsc4py(sparse_vector)
    row_start, row_end = vec.getOwnershipRange()
    out_size = len(dofs_list)
    out = OnlineVector(out_size)
    mpi_comm = vec.comm.tompi4py()
    for (index, dofs) in enumerate(dofs_list):
        assert len(dofs) == 1
        i = dofs[0]
        out_index = None
        vec_i_processor = -1
        if i >= row_start and i < row_end:
            out_index = vec.getValue(i)
            vec_i_processor = mpi_comm.rank
        vec_i_processor = mpi_comm.allreduce(vec_i_processor, op=MAX)
        assert vec_i_processor >= 0
        out[index] = mpi_comm.bcast(out_index, root=vec_i_processor)
    return out
