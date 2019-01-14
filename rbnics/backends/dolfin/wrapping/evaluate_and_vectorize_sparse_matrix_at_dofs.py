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
from rbnics.backends.online import OnlineVector
from rbnics.backends.dolfin.wrapping.to_petsc4py import to_petsc4py

def evaluate_and_vectorize_sparse_matrix_at_dofs(sparse_matrix, dofs_list):
    mat = to_petsc4py(sparse_matrix)
    row_start, row_end = mat.getOwnershipRange()
    out_size = len(dofs_list)
    out = OnlineVector(out_size)
    mpi_comm = mat.comm.tompi4py()
    for (index, dofs) in enumerate(dofs_list):
        assert len(dofs) == 2
        i = dofs[0]
        out_index = None
        mat_ij_processor = -1
        if i >= row_start and i < row_end:
            j = dofs[1]
            out_index = mat.getValue(i, j)
            mat_ij_processor = mpi_comm.rank
        mat_ij_processor = mpi_comm.allreduce(mat_ij_processor, op=MAX)
        assert mat_ij_processor >= 0
        out[index] = mpi_comm.bcast(out_index, root=mat_ij_processor)
    return out
