# Copyright (C) 2015-2021 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# This file contains a python translation of dolfin/io/XMLFunctionData.cpp private
# methods build_dof_map (renamed to _build_dof_map_reader_mapping) and build_global_to_cell_dof
# (renamed _build_dof_map_writer_mapping).
# These methods are going to be used when writing vectors and matrices
# to file. Please refer to the original file for original copyright information.
# The original implementation of _build_dof_map_reader_mapping, _build_dof_map_writer_mapping
# and _get_local_dofmap is
# Copyright (C) 2011 Garth N. Wells

from dolfin import cells
from rbnics.utils.cache import Cache


def build_dof_map_writer_mapping(V, local_dofmap=None):
    try:
        return _dof_map_writer_mapping_cache[V]
    except KeyError:
        def extract_first_cell(mapping_output):
            (min_global_cell_index, min_cell_dof) = mapping_output[0]
            for i in range(1, len(mapping_output)):
                (current_global_cell_index, current_cell_dof) = mapping_output[i]
                if current_global_cell_index < min_global_cell_index:
                    min_global_cell_index = current_global_cell_index
                    min_cell_dof = current_cell_dof
            return (min_global_cell_index, min_cell_dof)
        if local_dofmap is None:
            local_dofmap = _get_local_dofmap(V)
        dof_map_writer_mapping_original = _build_dof_map_writer_mapping(V, local_dofmap)
        dof_map_writer_mapping_storage = dict()
        for (key, value) in dof_map_writer_mapping_original.items():
            dof_map_writer_mapping_storage[key] = extract_first_cell(value)
        _dof_map_writer_mapping_cache[V] = dof_map_writer_mapping_storage
        return _dof_map_writer_mapping_cache[V]


_dof_map_writer_mapping_cache = Cache()


def build_dof_map_reader_mapping(V, local_dofmap=None):
    try:
        return _dof_map_reader_mapping_cache[V]
    except KeyError:
        if local_dofmap is None:
            local_dofmap = _get_local_dofmap(V)
        _dof_map_reader_mapping_cache[V] = _build_dof_map_reader_mapping(V, local_dofmap)
        return _dof_map_reader_mapping_cache[V]


_dof_map_reader_mapping_cache = Cache()


def _build_dof_map_writer_mapping(V, gathered_dofmap):  # was build_global_to_cell_dof in dolfin
    mpi_comm = V.mesh().mpi_comm()

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
                global_dof_to_cell_dof[gathered_dofmap[i]].append([global_cell_index, j])
                i += 1
    global_dof_to_cell_dof = mpi_comm.bcast(global_dof_to_cell_dof, root=0)
    return global_dof_to_cell_dof


def _build_dof_map_reader_mapping(V, gathered_dofmap):  # was build_dof_map in dolfin
    mesh = V.mesh()
    mpi_comm = mesh.mpi_comm()

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
    mpi_comm = mesh.mpi_comm()

    local_dofmap = list()  # of integers

    # Check that local-to-global cell numbering is available
    assert mesh.topology().have_global_indices(mesh.topology().dim())

    # Get local-to-global map
    local_to_global_dof = dofmap.tabulate_local_to_global_dofs()

    # Build dof map data with global cell indices
    for cell in cells(mesh):
        local_cell_index = cell.index()
        global_cell_index = cell.global_index()
        cell_dofs = dofmap.cell_dofs(local_cell_index)

        cell_dofs_global = list()
        for cell_dof in cell_dofs:
            cell_dofs_global.append(local_to_global_dof[cell_dof])

        # Store information as follows: global_cell_index, size of dofs, cell dof global 1, ...., cell dof global end
        local_dofmap.append(global_cell_index)
        local_dofmap.append(len(cell_dofs))
        local_dofmap.extend(cell_dofs_global)

    # Gather dof map data on root process
    gathered_dofmap = mpi_comm.gather(local_dofmap, root=0)
    if mpi_comm.rank == 0:
        gathered_dofmap_flattened = list()
        for proc_map in gathered_dofmap:
            gathered_dofmap_flattened.extend(proc_map)
        return gathered_dofmap_flattened
    else:
        return list()
