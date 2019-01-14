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
from rbnics.utils.cache import cache

def get_global_dof_coordinates(global_dof, V, global_to_local=None, local_dof_to_coordinates=None):
    if global_to_local is None:
        global_to_local = get_global_dof_to_local_dof_map(V, V.dofmap())
    if local_dof_to_coordinates is None:
        local_dof_to_coordinates = _get_local_dof_to_coordinates_map(V)
    
    mpi_comm = V.mesh().mpi_comm()
    dof_coordinates = None
    dof_coordinates_processor = -1
    if global_dof in global_to_local:
        dof_coordinates = local_dof_to_coordinates[global_to_local[global_dof]]
        dof_coordinates_processor = mpi_comm.rank
    dof_coordinates_processor = mpi_comm.allreduce(dof_coordinates_processor, op=MAX)
    assert dof_coordinates_processor >= 0
    return mpi_comm.bcast(dof_coordinates, root=dof_coordinates_processor)
    
@cache
def _get_local_dof_to_coordinates_map(V):
    return V.tabulate_dof_coordinates().reshape((-1, V.mesh().ufl_cell().topological_dimension()))
