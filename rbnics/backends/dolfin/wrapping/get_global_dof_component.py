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
