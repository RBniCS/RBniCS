# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

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
