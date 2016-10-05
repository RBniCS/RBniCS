# Copyright (C) 2015-2016 by the RBniCS authors
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
## @file functions_list.py
#  @brief Type for storing a list of FE functions.
#
#  @author Francesco Ballarin <francesco.ballarin@sissa.it>
#  @author Gianluigi Rozza    <gianluigi.rozza@sissa.it>
#  @author Alberto   Sartori  <alberto.sartori@sissa.it>

# This file contains a python translation of dolfin/io/XMLFunctionData.cpp private
# methods build_dof_map (renamed to build_dof_map_reader_mapping) and build_global_to_cell_dof
# (renamed build_dof_map_writer_mapping).
# These methods are going to be used when writing vectors and matrices
# to file. Please referer to the original file for original copyright information.
# The original implementation of _build_dof_map_writer_mapping and _get_local_dofmap is
# Copyright (C) 2011 Garth N. Wells

from dolfin import cells
from mpi4py.MPI import SUM

def build_dof_map_writer_mapping(V):
    def extract_first_cell(mapping_output):
        (min_global_cell_index, min_cell_dof) = mapping_output[0]
        for i in range(1, len(mapping_output)):
            (current_global_cell_index, current_cell_dof) = mapping_output[i]
            if current_global_cell_index < min_global_cell_index:
                min_global_cell_index = current_global_cell_index
                min_cell_dof = current_cell_dof
        return (min_global_cell_index, min_cell_dof)
    if not V in build_dof_map_writer_mapping._storage:
        dof_map_writer_mapping_original = _build_dof_map_writer_mapping(V)
        dof_map_writer_mapping_storage = dict()
        for (key, value) in dof_map_writer_mapping_original.iteritems():
            dof_map_writer_mapping_storage[key] = extract_first_cell(value)
        build_dof_map_writer_mapping._storage[V] = dof_map_writer_mapping_storage
    return build_dof_map_writer_mapping._storage[V]
build_dof_map_writer_mapping._storage = dict()

def build_dof_map_reader_mapping(V):
    if not V in build_dof_map_reader_mapping._storage:
        build_dof_map_reader_mapping._storage[V] = _build_dof_map_reader_mapping(V)
    return build_dof_map_reader_mapping._storage[V]
build_dof_map_reader_mapping._storage = dict()

def _build_dof_map_writer_mapping(V): # was build_global_to_cell_dof in dolfin
    dofmap = V.dofmap()
    mpi_comm = V.mesh().mpi_comm().tompi4py()
    gathered_dofmap = _get_local_dofmap(V)
    
    # Build global dof -> (global cell, local dof) map on root process
    global_dof_to_cell_dof = dict()
    if mpi_comm.rank == 0:
        i = 0
        while i < len(gathered_dofmap):
            global_cell_index = gathered_dofmap[i]
            i += 1
            num_dofs = gathered_dofmap[i]
            i += 1
            for j in range(num_dofs):
                if gathered_dofmap[i] not in global_dof_to_cell_dof:
                    global_dof_to_cell_dof[gathered_dofmap[i]] = list()
                global_dof_to_cell_dof[gathered_dofmap[i]].append( [global_cell_index, j] )
                i += 1
    global_dof_to_cell_dof = mpi_comm.bcast(global_dof_to_cell_dof, root=0)
    return global_dof_to_cell_dof
    
def _build_dof_map_reader_mapping(V): # was build_dof_map in dolfin
    mesh = V.mesh()
    mpi_comm = mesh.mpi_comm().tompi4py()
    gathered_dofmap = _get_local_dofmap(V)

    # Get global number of cells
    num_cells = mpi_comm.allreduce(mesh.num_cells(), op=SUM)
    
    # Build global dofmap on root process
    dof_map = dict()
    if mpi_comm.rank == 0:
        i = 0
        while i < len(gathered_dofmap):
            global_cell_index = gathered_dofmap[i]
            i += 1
            num_dofs = gathered_dofmap[i]
            i += 1
            assert global_cell_index not in dof_map
            dof_map[global_cell_index] = list()
            for j in range(num_dofs):
                dof_map[global_cell_index].append(gathered_dofmap[i])
                i += 1
    dof_map = mpi_comm.bcast(dof_map, root=0)
    return dof_map

def _get_local_dofmap(V):
    mesh = V.mesh()
    dofmap = V.dofmap()
    mpi_comm = mesh.mpi_comm().tompi4py()
    
    local_dofmap = list() # of integers
    if mpi_comm.size > 1:
        # Check that local-to-global cell numbering is available
        assert mesh.topology().have_global_indices(mesh.topology().dim())
        
        # Get local-to-global map
        local_to_global_dof = dofmap.tabulate_local_to_global_dofs()
        
        # Build dof map data with global cell indices
        for cell in cells(mesh):
            local_cell_index = cell.index()
            global_cell_index = cell.global_index()
            cell_dofs = dofmap.cell_dofs(local_cell_index)

            cell_dofs_global = list();
            for cell_dof in cell_dofs:
                cell_dofs_global.append(local_to_global_dof[cell_dof])
            
            # Store information as follows: global_cell_index, size of dofs, cell dof global 1, ...., cell dof global end
            local_dofmap.append(global_cell_index)
            local_dofmap.append(len(cell_dofs))
            local_dofmap.extend(cell_dofs_global)
    else:
        # Build dof map data with local cell indices
        for cell in cells(mesh):
            local_cell_index = cell.index()
            cell_dofs = dofmap.cell_dofs(local_cell_index)
            
            # Store information as follows: global_cell_index, size of dofs, cell dof global 1, ...., cell dof global end
            local_dofmap.append(local_cell_index)
            local_dofmap.append(len(cell_dofs))
            local_dofmap.extend(cell_dofs)

    # Gather dof map data on root process
    gathered_dofmap = mpi_comm.gather(local_dofmap, root=0)
    if mpi_comm.rank == 0:
        gathered_dofmap_flattened = list()
        for proc_map in gathered_dofmap:
            gathered_dofmap_flattened.extend(proc_map)
        return gathered_dofmap_flattened
    else:
        return None
    
