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

from logging import DEBUG, getLogger
from numpy import array, uintp, unique, where
from scipy.spatial.ckdtree import cKDTree as KDTree
from mpi4py.MPI import SUM
from dolfin import Cell, cells, compile_cpp_code, Facet, facets, FunctionSpace, Mesh, MeshEditor, MeshFunction, Vertex, vertices
from dolfin.cpp.mesh import MeshFunctionBool

logger = getLogger("rbnics/backends/dolfin/wrapping/create_submesh.py")

# Implement an extended version of cbcpost create_submesh that:
# a) as cbcpost version (and in contrast to standard dolfin) also works in parallel
# b) works for number of cells less than number of processors, by arbitrarily adding a cell on empty processors
# c) also assigns shared_entities(0). This is essential because otherwise mid-dofs on shared facets will be duplicated!
# d) assigns a global index of vertices and cells which is independent from the number of processors
# e) stores map between mesh and submesh cells, facets and vertices
# Part of this code is taken from cbcpost/utils/submesh.py, with the following copyright information:
# Copyright (C) 2010-2014 Simula Research Laboratory
def create_submesh(mesh, markers):
    mpi_comm = mesh.mpi_comm()
    assert isinstance(markers, MeshFunctionBool)
    assert markers.dim() == mesh.topology().dim()
    marker_id = True
    
    # == 1. Extract marked cells == #
    # Dolfin does not support a distributed mesh that is empty on some processes.
    # cbcpost gets around this by moving a single cell from the a non-empty processor to an empty one.
    # Note that, however, this cannot work if the number of marked cell is less than the number of processors.
    # In the interest of considering this case, we enable at least one cell (arbitrarily) on each processor.
    # We find this solution acceptable for our purposes, despite the increase of the reduced mesh size,
    # since we are never actually interested in solving a PDE on the reduced mesh, but rather only in
    # assemblying tensors on it and extract their values at some locations.
    backup_first_marker_id = None
    if marker_id not in markers.array():
        backup_first_marker_id = markers.array()[0]
        markers.array()[0] = marker_id
    assert marker_id in markers.array()
    
    # == 2. Create submesh == #
    submesh = Mesh(mesh.mpi_comm())
    mesh_editor = MeshEditor()
    mesh_editor.open(submesh,
                     mesh.ufl_cell().cellname(),
                     mesh.ufl_cell().topological_dimension(),
                     mesh.ufl_cell().geometric_dimension())
    # Extract cells from mesh with specified marker_id
    mesh_cell_indices = where(markers.array() == marker_id)[0]
    mesh_cells = mesh.cells()[mesh_cell_indices]
    mesh_global_cell_indices = sorted([mesh.topology().global_indices(mesh.topology().dim())[cell_index] for cell_index in mesh_cell_indices])
    # Get vertices of extracted cells
    mesh_vertex_indices = unique(mesh_cells.flatten())
    mesh_global_vertex_indices = sorted([mesh.topology().global_indices(0)[vertex_index] for vertex_index in mesh_vertex_indices])
    # Number vertices in a way which is independent from the number of processors. To do so ...
    # ... first of all collect all vertices from all processors
    allgathered_mesh_global_vertex_indices__non_empty_processors = list()
    allgathered_mesh_global_vertex_indices__empty_processors = list()
    for r in range(mpi_comm.size):
        backup_first_marker_id_r = mpi_comm.bcast(backup_first_marker_id, root=r)
        if backup_first_marker_id_r is None:
            allgathered_mesh_global_vertex_indices__non_empty_processors.extend(mpi_comm.bcast(mesh_global_vertex_indices, root=r))
        else:
            allgathered_mesh_global_vertex_indices__empty_processors.extend(mpi_comm.bcast(mesh_global_vertex_indices, root=r))
    allgathered_mesh_global_vertex_indices__non_empty_processors = sorted(unique(allgathered_mesh_global_vertex_indices__non_empty_processors))
    allgathered_mesh_global_vertex_indices__empty_processors = sorted(unique(allgathered_mesh_global_vertex_indices__empty_processors))
    # ... then create a dict that will contain the map from mesh global vertex index to submesh global vertex index.
    # ... Here make sure to number first "real" vertices (those coming from non empty processors), since the other ones
    # ... are just a side effect of the current partitioning!
    allgathered_mesh_to_submesh_vertex_global_indices = dict()
    _submesh_vertex_global_index = 0
    for mesh_vertex_global_index in allgathered_mesh_global_vertex_indices__non_empty_processors:
        assert mesh_vertex_global_index not in allgathered_mesh_to_submesh_vertex_global_indices
        allgathered_mesh_to_submesh_vertex_global_indices[mesh_vertex_global_index] = _submesh_vertex_global_index
        _submesh_vertex_global_index += 1
    for mesh_vertex_global_index in allgathered_mesh_global_vertex_indices__empty_processors:
        if mesh_vertex_global_index not in allgathered_mesh_to_submesh_vertex_global_indices:
            allgathered_mesh_to_submesh_vertex_global_indices[mesh_vertex_global_index] = _submesh_vertex_global_index
            _submesh_vertex_global_index += 1
    # Number cells in a way which is independent from the number of processors. To do so ...
    # ... first of all collect all cells from all processors
    allgathered_mesh_global_cell_indices__non_empty_processors = list()
    allgathered_mesh_global_cell_indices__empty_processors = list()
    for r in range(mpi_comm.size):
        backup_first_marker_id_r = mpi_comm.bcast(backup_first_marker_id, root=r)
        if backup_first_marker_id_r is None:
            allgathered_mesh_global_cell_indices__non_empty_processors.extend(mpi_comm.bcast(mesh_global_cell_indices, root=r))
        else:
            allgathered_mesh_global_cell_indices__empty_processors.extend(mpi_comm.bcast(mesh_global_cell_indices, root=r))
    allgathered_mesh_global_cell_indices__non_empty_processors = sorted(unique(allgathered_mesh_global_cell_indices__non_empty_processors))
    allgathered_mesh_global_cell_indices__empty_processors = sorted(unique(allgathered_mesh_global_cell_indices__empty_processors))
    # ... then create a dict that will contain the map from mesh global cell index to submesh global cell index.
    # ... Here make sure to number first "real" vertices (those coming from non empty processors), since the other ones
    # ... are just a side effect of the current partitioning!
    allgathered_mesh_to_submesh_cell_global_indices = dict()
    _submesh_cell_global_index = 0
    for mesh_cell_global_index in allgathered_mesh_global_cell_indices__non_empty_processors:
        assert mesh_cell_global_index not in allgathered_mesh_to_submesh_cell_global_indices
        allgathered_mesh_to_submesh_cell_global_indices[mesh_cell_global_index] = _submesh_cell_global_index
        _submesh_cell_global_index += 1
    for mesh_cell_global_index in allgathered_mesh_global_cell_indices__empty_processors:
        assert mesh_cell_global_index not in allgathered_mesh_to_submesh_cell_global_indices
        allgathered_mesh_to_submesh_cell_global_indices[mesh_cell_global_index] = _submesh_cell_global_index
        _submesh_cell_global_index += 1
    # Also create a mapping from mesh local vertex index to submesh local vertex index.
    mesh_to_submesh_vertex_local_indices = dict(zip(mesh_vertex_indices, list(range(len(mesh_vertex_indices)))))
    # Also create a mapping from mesh local cell index to submesh local cell index.
    mesh_to_submesh_cell_local_indices = dict(zip(mesh_cell_indices, list(range(len(mesh_cell_indices)))))
    # Now, define submesh cells
    submesh_cells = list()
    for i, c in enumerate(mesh_cells):
        submesh_cells.append([mesh_to_submesh_vertex_local_indices[j] for j in c])
    # Store vertices as submesh_vertices[local_index] = (global_index, coordinates)
    submesh_vertices = dict()
    for mesh_vertex_local_index, submesh_vertex_local_index in mesh_to_submesh_vertex_local_indices.items():
        submesh_vertices[submesh_vertex_local_index] = (
            allgathered_mesh_to_submesh_vertex_global_indices[mesh.topology().global_indices(0)[mesh_vertex_local_index]],
            mesh.coordinates()[mesh_vertex_local_index]
        )
    # Collect the global number of vertices and cells
    global_num_cells = mpi_comm.allreduce(len(submesh_cells), op=SUM)
    global_num_vertices = len(allgathered_mesh_to_submesh_vertex_global_indices)
    # Fill in mesh_editor
    mesh_editor.init_vertices_global(len(submesh_vertices), global_num_vertices)
    mesh_editor.init_cells_global(len(submesh_cells), global_num_cells)
    for local_index, cell_vertices in enumerate(submesh_cells):
        mesh_editor.add_cell(local_index, cell_vertices)
    for local_index, (global_index, coordinates) in submesh_vertices.items():
        mesh_editor.add_vertex_global(local_index, global_index, coordinates)
    mesh_editor.close()
    # Initialize topology
    submesh.topology().init(0, len(submesh_vertices), global_num_vertices)
    submesh.topology().init(mesh.ufl_cell().topological_dimension(), len(submesh_cells), global_num_cells)
    # Correct the global index of cells
    for local_index in range(len(submesh_cells)):
        submesh.topology().set_global_index(
            submesh.topology().dim(),
            local_index,
            allgathered_mesh_to_submesh_cell_global_indices[mesh_global_cell_indices[local_index]]
        )
    
    # == 3. Store (local) mesh to/from submesh map for cells, facets and vertices == #
    # Cells
    submesh.mesh_to_submesh_cell_local_indices = mesh_to_submesh_cell_local_indices
    submesh.submesh_to_mesh_cell_local_indices = mesh_cell_indices
    # Vertices
    submesh.mesh_to_submesh_vertex_local_indices = mesh_to_submesh_vertex_local_indices
    submesh.submesh_to_mesh_vertex_local_indices = mesh_vertex_indices
    # Facets
    mesh_vertices_to_mesh_facets = dict()
    mesh_facets_to_mesh_vertices = dict()
    for mesh_cell_index in mesh_cell_indices:
        mesh_cell = Cell(mesh, mesh_cell_index)
        for mesh_facet in facets(mesh_cell):
            mesh_facet_vertices = list()
            for mesh_facet_vertex in vertices(mesh_facet):
                mesh_facet_vertices.append(mesh_facet_vertex.index())
            mesh_facet_vertices = tuple(sorted(mesh_facet_vertices))
            if mesh_facet_vertices in mesh_vertices_to_mesh_facets:
                assert mesh_vertices_to_mesh_facets[mesh_facet_vertices] == mesh_facet.index()
            else:
                mesh_vertices_to_mesh_facets[mesh_facet_vertices] = mesh_facet.index()
            if mesh_facet.index() in mesh_facets_to_mesh_vertices:
                assert mesh_facets_to_mesh_vertices[mesh_facet.index()] == mesh_facet_vertices
            else:
                mesh_facets_to_mesh_vertices[mesh_facet.index()] = mesh_facet_vertices
    submesh_vertices_to_submesh_facets = dict()
    submesh_facets_to_submesh_vertices = dict()
    for submesh_facet in facets(submesh):
        submesh_facet_vertices = list()
        for submesh_facet_vertex in vertices(submesh_facet):
            submesh_facet_vertices.append(submesh_facet_vertex.index())
        submesh_facet_vertices = tuple(sorted(submesh_facet_vertices))
        assert submesh_facet_vertices not in submesh_vertices_to_submesh_facets
        submesh_vertices_to_submesh_facets[submesh_facet_vertices] = submesh_facet.index()
        assert submesh_facet.index() not in submesh_facets_to_submesh_vertices
        submesh_facets_to_submesh_vertices[submesh_facet.index()] = submesh_facet_vertices
    mesh_to_submesh_facets_local_indices = dict()
    for (mesh_facet_index, mesh_vertices) in mesh_facets_to_mesh_vertices.items():
        submesh_vertices = tuple(sorted([submesh.mesh_to_submesh_vertex_local_indices[mesh_vertex] for mesh_vertex in mesh_vertices]))
        submesh_facet_index = submesh_vertices_to_submesh_facets[submesh_vertices]
        mesh_to_submesh_facets_local_indices[mesh_facet_index] = submesh_facet_index
    submesh_to_mesh_facets_local_indices = dict()
    for (submesh_facet_index, submesh_vertices) in submesh_facets_to_submesh_vertices.items():
        mesh_vertices = tuple(sorted([submesh.submesh_to_mesh_vertex_local_indices[submesh_vertex] for submesh_vertex in submesh_vertices]))
        mesh_facet_index = mesh_vertices_to_mesh_facets[mesh_vertices]
        submesh_to_mesh_facets_local_indices[submesh_facet_index] = mesh_facet_index
    submesh.mesh_to_submesh_facet_local_indices = mesh_to_submesh_facets_local_indices
    submesh.submesh_to_mesh_facet_local_indices = list()
    assert min(submesh_to_mesh_facets_local_indices.keys()) == 0
    assert max(submesh_to_mesh_facets_local_indices.keys()) == len(submesh_to_mesh_facets_local_indices.keys()) - 1
    for submesh_facet_index in range(len(submesh_to_mesh_facets_local_indices)):
        submesh.submesh_to_mesh_facet_local_indices.append(submesh_to_mesh_facets_local_indices[submesh_facet_index])
    # == 3bis. Prepare (temporary) global indices of facets == #
    # Wrapper to DistributedMeshTools::number_entities
    cpp_code = """
        #include <pybind11/pybind11.h>
        #include <dolfin/mesh/DistributedMeshTools.h>
        #include <dolfin/mesh/Mesh.h>
        
        void initialize_global_indices(std::shared_ptr<dolfin::Mesh> mesh, std::size_t dim)
        {
            dolfin::DistributedMeshTools::number_entities(*mesh, dim);
        }
        
        PYBIND11_MODULE(SIGNATURE, m)
        {
            m.def("initialize_global_indices", &initialize_global_indices);
        }
    """
    initialize_global_indices = compile_cpp_code(cpp_code).initialize_global_indices
    initialize_global_indices(mesh, mesh.topology().dim() - 1)
    # Prepare global indices of facets
    mesh_facets_local_to_global_indices = dict()
    for mesh_cell_index in mesh_cell_indices:
        mesh_cell = Cell(mesh, mesh_cell_index)
        for mesh_facet in facets(mesh_cell):
            mesh_facets_local_to_global_indices[mesh_facet.index()] = mesh_facet.global_index()
    mesh_facets_global_indices_in_submesh = list()
    for mesh_facet_local_index in mesh_to_submesh_facets_local_indices.keys():
        mesh_facets_global_indices_in_submesh.append(mesh_facets_local_to_global_indices[mesh_facet_local_index])
    allgathered__mesh_facets_global_indices_in_submesh = list()
    for r in range(mpi_comm.size):
        allgathered__mesh_facets_global_indices_in_submesh.extend(mpi_comm.bcast(mesh_facets_global_indices_in_submesh, root=r))
    allgathered__mesh_facets_global_indices_in_submesh = sorted(set(allgathered__mesh_facets_global_indices_in_submesh))
    mesh_to_submesh_facets_global_indices = dict()
    for (submesh_facet_global_index, mesh_facet_global_index) in enumerate(allgathered__mesh_facets_global_indices_in_submesh):
        mesh_to_submesh_facets_global_indices[mesh_facet_global_index] = submesh_facet_global_index
    submesh_facets_local_to_global_indices = dict()
    for (submesh_facet_local_index, mesh_facet_local_index) in submesh_to_mesh_facets_local_indices.items():
        submesh_facets_local_to_global_indices[submesh_facet_local_index] = mesh_to_submesh_facets_global_indices[mesh_facets_local_to_global_indices[mesh_facet_local_index]]
    
    # == 4. Assign shared vertices == #
    shared_entities_dimensions = {
        "vertex": 0,
        "facet": submesh.topology().dim() - 1,
        "cell": submesh.topology().dim()
    }
    shared_entities_class = {
        "vertex": Vertex,
        "facet": Facet,
        "cell": Cell
    }
    shared_entities_iterator = {
        "vertex": vertices,
        "facet": facets,
        "cell": cells
    }
    shared_entities_submesh_global_index_getter = {
        "vertex": lambda entity: entity.global_index(),
        "facet": lambda entity: submesh_facets_local_to_global_indices[entity.index()],
        "cell": lambda entity: entity.global_index()
    }
    for entity_type in ["vertex", "facet", "cell"]: # do not use .keys() because the order is important
        dim = shared_entities_dimensions[entity_type]
        class_ = shared_entities_class[entity_type]
        iterator = shared_entities_iterator[entity_type]
        submesh_global_index_getter = shared_entities_submesh_global_index_getter[entity_type]
        # Get shared entities from mesh. A subset of these will end being shared entities also the submesh
        # (thanks to the fact that we do not redistribute cells from one processor to another)
        if mpi_comm.size > 1: # some entities may not be initialized in serial, since they are not needed
            assert mesh.topology().have_shared_entities(dim), "Mesh shared entities have not been initialized for dimension " + str(dim)
        if mesh.topology().have_shared_entities(dim): # always true in parallel (when really needed)
            # However, it may happen that an entity which has been selected is not shared anymore because only one of
            # the sharing processes has it in the submesh. For instance, consider the case
            # of two cells across the interface (located on a facet f) between two processors. It may happen that
            # only one of the two cells is selected: the facet f and its vertices are not shared anymore!
            # For this reason, we create a new dict from global entity index to processors sharing them. Thus ...
            # ... first of all get global indices corresponding to local entities
            if entity_type in ["vertex", "cell"]:
                assert submesh.topology().have_global_indices(dim), "Submesh global indices have not been initialized for dimension " + str(dim)
            submesh_local_entities_global_index = list()
            submesh_local_entities_global_to_local_index = dict()
            for entity in iterator(submesh):
                local_entity_index = entity.index()
                global_entity_index = submesh_global_index_getter(entity)
                submesh_local_entities_global_index.append(global_entity_index)
                submesh_local_entities_global_to_local_index[global_entity_index] = local_entity_index
            # ... then gather all global indices from all processors
            gathered__submesh_local_entities_global_index = list() # over processor id
            for r in range(mpi_comm.size):
                gathered__submesh_local_entities_global_index.append(mpi_comm.bcast(submesh_local_entities_global_index, root=r))
            # ... then create dict from global index to processors sharing it
            submesh_shared_entities__global = dict()
            for r in range(mpi_comm.size):
                for global_entity_index in gathered__submesh_local_entities_global_index[r]:
                    if global_entity_index not in submesh_shared_entities__global:
                        submesh_shared_entities__global[global_entity_index] = list()
                    submesh_shared_entities__global[global_entity_index].append(r)
            # ... and finally popuplate shared entities dict, which is the same as the dict above except that
            # the current processor rank is removed and a local indexing is used
            submesh_shared_entities = dict() # from local index to list of integers
            for (global_entity_index, processors) in submesh_shared_entities__global.items():
                if (
                    mpi_comm.rank in processors  # only local entities
                        and
                    len(processors) > 1 # it was still shared after submesh extraction
                ):
                    other_processors_list = list(processors)
                    other_processors_list.remove(mpi_comm.rank)
                    other_processors = array(other_processors_list, dtype=uintp)
                    submesh_shared_entities[submesh_local_entities_global_to_local_index[global_entity_index]] = other_processors

            # Need an extension module to populate shared_entities because in python each call to shared_entities
            # returns a temporary.
            cpp_code = """
                #include <Eigen/Core>
                #include <pybind11/pybind11.h>
                #include <pybind11/eigen.h>
                #include <dolfin/mesh/Mesh.h>
                
                using OtherProcesses = Eigen::Ref<const Eigen::Matrix<std::size_t, Eigen::Dynamic, 1>>;
                
                void set_shared_entities(std::shared_ptr<dolfin::Mesh> submesh, std::size_t idx, const OtherProcesses other_processes, std::size_t dim)
                {
                    std::set<unsigned int> set_other_processes;
                    for (std::size_t i(0); i < other_processes.size(); i++)
                        set_other_processes.insert(other_processes[i]);
                    submesh->topology().shared_entities(dim)[idx] = set_other_processes;
                }
                
                PYBIND11_MODULE(SIGNATURE, m)
                {
                    m.def("set_shared_entities", &set_shared_entities);
                }
            """
            set_shared_entities = compile_cpp_code(cpp_code).set_shared_entities
            for (submesh_entity_local_index, other_processors) in submesh_shared_entities.items():
                set_shared_entities(submesh, submesh_entity_local_index, other_processors, dim)
                
            logger.log(DEBUG, "Local indices of shared entities for dimension " + str(dim) + ": " + str(list(submesh.topology().shared_entities(0).keys())))
            logger.log(DEBUG, "Global indices of shared entities for dimension " + str(dim) + ": " + str([class_(submesh, local_index).global_index() for local_index in submesh.topology().shared_entities(dim).keys()]))
    
    # == 5. Also initialize submesh facets global indices, now that shared facets have been computed == #
    initialize_global_indices(submesh, submesh.topology().dim() - 1) # note that DOLFIN might change the numbering when compared to the one at 3bis
    
    # == 6. Restore backup_first_marker_id and return == #
    if backup_first_marker_id is not None:
        markers.array()[0] = backup_first_marker_id
    return submesh

def convert_meshfunctions_to_submesh(mesh, submesh, meshfunctions_on_mesh):
    assert meshfunctions_on_mesh is None or (isinstance(meshfunctions_on_mesh, list) and len(meshfunctions_on_mesh) > 0)
    if meshfunctions_on_mesh is None:
        return None
    meshfunctions_on_submesh = list()
    # Create submesh subdomains
    for mesh_subdomain in meshfunctions_on_mesh:
        submesh_subdomain = MeshFunction("size_t", submesh, mesh_subdomain.dim())
        submesh_subdomain.set_all(0)
        assert submesh_subdomain.dim() in (submesh.topology().dim(), submesh.topology().dim() - 1)
        if submesh_subdomain.dim() == submesh.topology().dim():
            for submesh_cell in cells(submesh):
                submesh_subdomain.array()[submesh_cell.index()] = mesh_subdomain.array()[submesh.submesh_to_mesh_cell_local_indices[submesh_cell.index()]]
        elif submesh_subdomain.dim() == submesh.topology().dim() - 1:
            for submesh_facet in facets(submesh):
                submesh_subdomain.array()[submesh_facet.index()] = mesh_subdomain.array()[submesh.submesh_to_mesh_facet_local_indices[submesh_facet.index()]]
        else: # impossible to arrive here anyway, thanks to the assert
            raise TypeError("Invalid arguments in convert_meshfunctions_to_submesh.")
        meshfunctions_on_submesh.append(submesh_subdomain)
    return meshfunctions_on_submesh
    
def convert_functionspace_to_submesh(functionspace_on_mesh, submesh, CustomFunctionSpace=None):
    if CustomFunctionSpace is None:
        CustomFunctionSpace = FunctionSpace
    functionspace_on_submesh = CustomFunctionSpace(submesh, functionspace_on_mesh.ufl_element())
    return functionspace_on_submesh
    
# This function is similar to cbcpost restriction_map. The main difference are:
# a) it builds a KDTree for each cell, rather than exploring the entire submesh at a time, so that
#    no ambiguity should arise even in DG function spaces
# b) the maps from/to dofs are with respect to global dof indices
# Part of this code is taken from cbcpost/utils/restriction_map.py, with the following copyright information:
# Copyright (C) 2010-2014 Simula Research Laboratory
def map_functionspaces_between_mesh_and_submesh(functionspace_on_mesh, mesh, functionspace_on_submesh, submesh, global_indices=True):
    mesh_dofs_to_submesh_dofs = dict()
    submesh_dofs_to_mesh_dofs = dict()
    
    # Initialize map from mesh dofs to submesh dofs, and viceversa
    if functionspace_on_mesh.num_sub_spaces() > 0:
        assert functionspace_on_mesh.num_sub_spaces() == functionspace_on_submesh.num_sub_spaces()
        for i in range(functionspace_on_mesh.num_sub_spaces()):
            (mesh_dofs_to_submesh_dofs_i, submesh_dofs_to_mesh_dofs_i) = map_functionspaces_between_mesh_and_submesh(functionspace_on_mesh.sub(i), mesh, functionspace_on_submesh.sub(i), submesh, global_indices)
            for (mesh_dof, submesh_dof) in mesh_dofs_to_submesh_dofs_i.items():
                assert mesh_dof not in mesh_dofs_to_submesh_dofs
                assert submesh_dof not in submesh_dofs_to_mesh_dofs
            mesh_dofs_to_submesh_dofs.update(mesh_dofs_to_submesh_dofs_i)
            submesh_dofs_to_mesh_dofs.update(submesh_dofs_to_mesh_dofs_i)
        # Return
        return (mesh_dofs_to_submesh_dofs, submesh_dofs_to_mesh_dofs)
    else:
        assert functionspace_on_mesh.ufl_element().family() in ("Lagrange", "Discontinuous Lagrange"), "The current implementation has been tested only for Lagrange or Discontinuous Lagrange function spaces"
        assert functionspace_on_submesh.ufl_element().family() in ("Lagrange", "Discontinuous Lagrange"), "The current implementation has been tested only for Lagrange or Discontinuous Lagrange function spaces"
        mesh_element = functionspace_on_mesh.element()
        mesh_dofmap = functionspace_on_mesh.dofmap()
        submesh_element = functionspace_on_submesh.element()
        submesh_dofmap = functionspace_on_submesh.dofmap()
        for submesh_cell in cells(submesh):
            submesh_dof_coordinates = submesh_element.tabulate_dof_coordinates(submesh_cell)
            submesh_cell_dofs = submesh_dofmap.cell_dofs(submesh_cell.index())
            if global_indices:
                submesh_cell_dofs = [functionspace_on_submesh.dofmap().local_to_global_index(local_dof) for local_dof in submesh_cell_dofs]
            mesh_cell = Cell(mesh, submesh.submesh_to_mesh_cell_local_indices[submesh_cell.index()])
            mesh_dof_coordinates = mesh_element.tabulate_dof_coordinates(mesh_cell)
            mesh_cell_dofs = mesh_dofmap.cell_dofs(mesh_cell.index())
            if global_indices:
                mesh_cell_dofs = [functionspace_on_mesh.dofmap().local_to_global_index(local_dof) for local_dof in mesh_cell_dofs]
            assert len(submesh_dof_coordinates) == len(mesh_dof_coordinates)
            assert len(submesh_cell_dofs) == len(mesh_cell_dofs)
            # Build a KDTree to compute distances from coordinates in mesh
            kdtree = KDTree(mesh_dof_coordinates)
            distances, mesh_indices = kdtree.query(submesh_dof_coordinates)
            # Map from mesh to submesh
            for (i, submesh_dof) in enumerate(submesh_cell_dofs):
                distance, mesh_index = distances[i], mesh_indices[i]
                assert distance < mesh_cell.h()*1e-5
                mesh_dof = mesh_cell_dofs[mesh_index]
                if mesh_dof not in mesh_dofs_to_submesh_dofs:
                    mesh_dofs_to_submesh_dofs[mesh_dof] = submesh_dof
                else:
                    assert mesh_dofs_to_submesh_dofs[mesh_dof] == submesh_dof
                if submesh_dof not in submesh_dofs_to_mesh_dofs:
                    submesh_dofs_to_mesh_dofs[submesh_dof] = mesh_dof
                else:
                    assert submesh_dofs_to_mesh_dofs[submesh_dof] == mesh_dof
        # Broadcast in parallel
        if global_indices:
            mpi_comm = mesh.mpi_comm()
            allgathered_mesh_dofs_to_submesh_dofs = mpi_comm.bcast(mesh_dofs_to_submesh_dofs, root=0)
            allgathered_submesh_dofs_to_mesh_dofs = mpi_comm.bcast(submesh_dofs_to_mesh_dofs, root=0)
            for r in range(1, mpi_comm.size):
                allgathered_mesh_dofs_to_submesh_dofs.update(mpi_comm.bcast(mesh_dofs_to_submesh_dofs, root=r))
                allgathered_submesh_dofs_to_mesh_dofs.update(mpi_comm.bcast(submesh_dofs_to_mesh_dofs, root=r))
        else:
            allgathered_mesh_dofs_to_submesh_dofs = mesh_dofs_to_submesh_dofs
            allgathered_submesh_dofs_to_mesh_dofs = submesh_dofs_to_mesh_dofs
        # Return
        return (allgathered_mesh_dofs_to_submesh_dofs, allgathered_submesh_dofs_to_mesh_dofs)
