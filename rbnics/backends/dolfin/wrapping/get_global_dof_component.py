# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from mpi4py.MPI import MAX
from rbnics.backends.dolfin.wrapping.get_global_dof_to_local_dof_map import get_global_dof_to_local_dof_map
from rbnics.backends.dolfin.wrapping.get_local_dof_to_component_map import get_local_dof_to_component_map


def get_global_dof_component(global_dof, V, global_to_local=None, local_dof_to_component=None):
    if global_to_local is None:
        global_to_local = get_global_dof_to_local_dof_map(V, V.dofmap())
    if local_dof_to_component is None:
        local_dof_to_component = get_local_dof_to_component_map(V)

    mpi_comm = V.mesh().mpi_comm()
    dof_component = None
    dof_component_processor = -1
    if global_dof in global_to_local:
        dof_component = local_dof_to_component[global_to_local[global_dof]]
        dof_component_processor = mpi_comm.rank
    dof_component_processor = mpi_comm.allreduce(dof_component_processor, op=MAX)
    assert dof_component_processor >= 0
    return mpi_comm.bcast(dof_component, root=dof_component_processor)
