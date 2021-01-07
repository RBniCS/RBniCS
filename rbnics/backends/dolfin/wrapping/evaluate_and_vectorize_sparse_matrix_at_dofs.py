# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
